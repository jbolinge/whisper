# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

WhisperX Audio Transcription with Speaker Diarization - a self-hosted Gradio web application for transcribing audio files with automatic speaker identification. Optimized for CPU-only environments using int8 quantization.

## Common Commands

```bash
# Setup (first time)
./setup.sh

# Or manually sync dependencies
uv sync --python 3.11

# Run the application
uv run python app.py

# Verify installation
uv run python -c "import whisperx; import gradio; print('Installation successful!')"

# Add a new dependency
uv add <package-name>
```

The server runs at `http://localhost:7860`.

## Dependency Management

Dependencies are managed via `pyproject.toml` using uv. PyTorch CPU is sourced from the dedicated pytorch-cpu index, and WhisperX is installed from git.

## Architecture

**Single-file application** (`app.py`) with these main components:

- `transcribe_audio()` - Main transcription pipeline:
  1. Load WhisperX model (CPU/CUDA auto-detect)
  2. Transcribe audio with whisperx
  3. Align transcript for word-level timestamps
  4. Perform speaker diarization (if HF token provided)
  5. Format output with speaker labels and timestamps

- `create_ui()` - Gradio interface configuration, handles HF token from `.env` or UI input

- `format_transcript_with_speakers()` / `format_transcript_simple()` - Output formatters that group consecutive segments by speaker

**Key dependencies:**
- WhisperX (faster-whisper based) for transcription
- pyannote-audio for speaker diarization (requires HuggingFace token)
- Gradio 4.x for web UI

## Environment Configuration

HuggingFace token can be provided via:
1. `.env` file with `HF_TOKEN` or `HUGGINGFACE_TOKEN`
2. UI text input (takes precedence over .env)

Speaker diarization requires accepting licenses at:
- https://huggingface.co/pyannote/speaker-diarization-3.1
- https://huggingface.co/pyannote/segmentation-3.0
