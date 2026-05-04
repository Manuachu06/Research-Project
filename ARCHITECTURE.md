# Multi-Agent Image-to-Scene System for Music Generation

## Goal
Convert an image into a rich scene description that can drive music generation,
including emotions, cinematic cues, and structured music controls.

## Agent Design
1. **VisionAnalystAgent**
   - Uses a vision detection tool and captioning tool.
   - Extracts objects, lighting, weather, composition, colors, and motion.

2. **EmotionCuratorAgent**
   - Infers weighted emotions (e.g., serenity, awe, melancholy, tension).
   - Can be backed by a classifier, LLM, or multimodal model.

3. **SceneComposerAgent**
   - Merges caption + visual signals + emotion profile.
   - Produces a compact, music-aware scene narrative.

4. **MusicPromptAgent**
   - Translates scene/emotions into music controls:
     - tempo
     - mode/scale
     - instrumentation
     - energy curve
     - generation prompt

5. **Orchestrator**
   - Runs agents in sequence and stores state in a shared schema.

## Tooling Strategy
- `VisionTool`: object/attribute extraction.
- `CaptionTool`: natural-language scene caption.
- `EmotionTool`: emotion inference probabilities.

The project includes mock tools for baseline testing and easy replacement with
real APIs/models.

## Output Schema
```json
{
  "image_path": "...",
  "visual_data": {"objects": [], "lighting": "", "weather": "", "colors": []},
  "caption": "...",
  "emotions": {"serenity": 0.78, "awe": 0.64},
  "scene_description": "...",
  "music_controls": {
    "tempo_bpm": 68,
    "mode": "Lydian",
    "instrumentation": ["soft piano"],
    "generation_prompt": "..."
  }
}
```

## How this helps music generation
- Encodes image semantics into emotional and musical latent controls.
- Produces deterministic structured controls for downstream music models.
- Supports prompt engineering and controllable generation workflows.
