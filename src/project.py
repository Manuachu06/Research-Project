"""Multi-agent pipeline for turning an image into music-oriented scene descriptions.

The pipeline is designed for research projects where scene understanding from images
is transformed into structured prompts for downstream music generation models.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Protocol, Any
import json


# ---------- Tool layer ----------
class VisionTool(Protocol):
    def detect(self, image_path: str) -> Dict[str, Any]:
        """Return visual analysis data extracted from an image."""


class CaptionTool(Protocol):
    def caption(self, image_path: str) -> str:
        """Return a natural-language caption for an image."""


class EmotionTool(Protocol):
    def infer(self, caption: str, visual_data: Dict[str, Any]) -> Dict[str, float]:
        """Infer emotion probabilities from textual + visual cues."""


class MockVisionTool:
    """Mock vision model for local development and testing."""

    def detect(self, image_path: str) -> Dict[str, Any]:
        return {
            "objects": ["person", "lake", "mountains", "sunset"],
            "lighting": "warm golden hour",
            "weather": "clear",
            "composition": "wide shot",
            "colors": ["orange", "purple", "deep blue"],
            "motion": "calm",
        }


class MockCaptionTool:
    def caption(self, image_path: str) -> str:
        return (
            "A lone person stands by a calm lake at sunset, facing distant mountains "
            "under warm orange and purple skies."
        )


class HeuristicEmotionTool:
    """Simple emotion inference baseline; replace with an LLM or classifier."""

    def infer(self, caption: str, visual_data: Dict[str, Any]) -> Dict[str, float]:
        emotions = {
            "serenity": 0.78,
            "awe": 0.64,
            "melancholy": 0.33,
            "joy": 0.41,
            "tension": 0.08,
        }
        if "storm" in caption.lower() or visual_data.get("weather") == "stormy":
            emotions["tension"] = 0.72
            emotions["serenity"] = 0.21
        return emotions


# ---------- Shared memory ----------
@dataclass
class SceneState:
    image_path: str
    visual_data: Dict[str, Any]
    caption: str
    emotions: Dict[str, float]
    scene_description: str
    music_controls: Dict[str, Any]


# ---------- Agents ----------
class VisionAnalystAgent:
    def __init__(self, vision_tool: VisionTool, caption_tool: CaptionTool):
        self.vision_tool = vision_tool
        self.caption_tool = caption_tool

    def run(self, image_path: str) -> Dict[str, Any]:
        return {
            "visual_data": self.vision_tool.detect(image_path),
            "caption": self.caption_tool.caption(image_path),
        }


class EmotionCuratorAgent:
    def __init__(self, emotion_tool: EmotionTool):
        self.emotion_tool = emotion_tool

    def run(self, caption: str, visual_data: Dict[str, Any]) -> Dict[str, float]:
        return self.emotion_tool.infer(caption, visual_data)


class SceneComposerAgent:
    def run(self, caption: str, visual_data: Dict[str, Any], emotions: Dict[str, float]) -> str:
        top_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:3]
        emo_text = ", ".join(f"{k} ({v:.2f})" for k, v in top_emotions)
        return (
            f"Scene: {caption}\\n"
            f"Visual style: {visual_data.get('composition')} with {visual_data.get('lighting')} light; "
            f"palette {', '.join(visual_data.get('colors', []))}.\\n"
            f"Dominant emotions: {emo_text}.\\n"
            "Musical implication: begin sparse and reflective, then add gentle harmonic lift "
            "to preserve calm while hinting at wonder."
        )


class MusicPromptAgent:
    def run(self, visual_data: Dict[str, Any], emotions: Dict[str, float]) -> Dict[str, Any]:
        serenity = emotions.get("serenity", 0.0)
        awe = emotions.get("awe", 0.0)
        tension = emotions.get("tension", 0.0)

        tempo = 68 if serenity > 0.6 else 84
        mode = "Lydian" if awe > 0.5 else "Major"
        if tension > 0.6:
            mode = "Phrygian"
            tempo = 96

        return {
            "tempo_bpm": tempo,
            "mode": mode,
            "energy_curve": ["low", "medium-low", "medium"],
            "instrumentation": ["soft piano", "warm pads", "light strings"],
            "texture": visual_data.get("motion", "steady"),
            "generation_prompt": (
                "Create cinematic ambient music aligned with extracted scene emotions. "
                f"Tempo {tempo} BPM, mode {mode}, evolving from introspection to gentle uplift."
            ),
        }


class Orchestrator:
    def __init__(self):
        self.vision_agent = VisionAnalystAgent(MockVisionTool(), MockCaptionTool())
        self.emotion_agent = EmotionCuratorAgent(HeuristicEmotionTool())
        self.scene_agent = SceneComposerAgent()
        self.music_agent = MusicPromptAgent()

    def run(self, image_path: str) -> SceneState:
        vision_out = self.vision_agent.run(image_path)
        emotions = self.emotion_agent.run(vision_out["caption"], vision_out["visual_data"])
        scene_description = self.scene_agent.run(vision_out["caption"], vision_out["visual_data"], emotions)
        music_controls = self.music_agent.run(vision_out["visual_data"], emotions)

        return SceneState(
            image_path=image_path,
            visual_data=vision_out["visual_data"],
            caption=vision_out["caption"],
            emotions=emotions,
            scene_description=scene_description,
            music_controls=music_controls,
        )


def run_pipeline(image_path: str) -> Dict[str, Any]:
    state = Orchestrator().run(image_path)
    return asdict(state)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Image-to-scene multi-agent pipeline for music generation.")
    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    args = parser.parse_args()

    result = run_pipeline(args.image_path)
    if args.pretty:
        print(json.dumps(result, indent=2))
    else:
        print(json.dumps(result))
