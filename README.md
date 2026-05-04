# Research-Project

A multi-agent research prototype that converts an image into a **scene description**
optimized for **music generation**, including explicit **emotion modeling**.

## Features
- Multi-agent pipeline design.
- Tool-driven scene extraction (vision, caption, emotion).
- Structured scene description for music systems.
- Music controls such as tempo, mode, instrumentation, and energy curve.

## Project structure
- `src/project.py`: main implementation (agents, tools, orchestration).
- `ARCHITECTURE.md`: system design and data flow.

## Run
```bash
python3 src/project.py /path/to/image.jpg --pretty
```

## Example output includes
- Caption and visual attributes.
- Emotion probabilities.
- Music-oriented scene narrative.
- Final generation prompt and musical controls.
