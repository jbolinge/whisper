# 🎙️ WhisperX Audio Transcription with Speaker Diarization

A self-hosted web application for transcribing audio files with automatic speaker identification. Built with WhisperX and Gradio, optimized for CPU-only environments.

## Features

- **Accurate Transcription**: Uses WhisperX (based on faster-whisper) for high-quality speech-to-text
- **Speaker Diarization**: Automatically identifies and labels different speakers
- **Long Audio Support**: Handles files up to 3+ hours
- **Web Interface**: Easy-to-use Gradio UI for uploading and downloading
- **CPU Optimized**: Runs efficiently on CPU with int8 quantization
- **Plain Text Output**: Clean, readable transcripts with timestamps

## Prerequisites

### System Requirements

- **OS**: Linux (Ubuntu 20.04+ recommended)
- **RAM**: 16GB minimum, 32GB+ recommended for large files
- **CPU**: Multi-core processor (more cores = faster processing)
- **Storage**: At least 10GB free (for models and temporary files)

### Required Software

**FFmpeg** (for audio processing):
```bash
sudo apt update
sudo apt install ffmpeg
```

### HuggingFace Account (for Speaker Diarization)

Speaker diarization requires access to pyannote models:

1. Create a [HuggingFace account](https://huggingface.co/join)
2. Go to [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
3. Accept the license agreement
4. Go to [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
5. Accept the license agreement
6. Create an access token at [HuggingFace Settings > Tokens](https://huggingface.co/settings/tokens)
7. Save your token - you'll need it when using the app

### Configuring Your HuggingFace Token

You have two options for providing your HuggingFace token:

**Option A: Using a `.env` file (Recommended)**

Create a `.env` file in the project directory:

```bash
cp .env.example .env
nano .env  # or use your preferred editor
```

Add your token:
```
HF_TOKEN=hf_your_token_here
```

The application will automatically load this token on startup. The UI will show a green checkmark indicating the token is loaded.

**Option B: Paste in the UI**

If you don't want to store the token in a file, you can paste it directly into the web interface each time you use it.

> **Note:** If both are provided, the UI input takes precedence over the `.env` file.

## Installation

### Option 1: Automated Setup (Recommended)

The setup script will install `uv` (if needed) and all dependencies:

```bash
# Clone or download the project
mkdir whisper-transcriber
cd whisper-transcriber
# (copy all project files here)

# Run automated setup
chmod +x setup.sh
./setup.sh
```

### Option 2: Manual Installation with uv

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc  # or restart your shell

# Create project directory
mkdir whisper-transcriber
cd whisper-transcriber
# (copy all project files here)

# Install all dependencies (creates venv automatically)
uv sync --python 3.11
```

### Verify Installation

```bash
uv run python -c "import whisperx; import gradio; print('Installation successful!')"
```

## Usage

### Starting the Server

```bash
uv run python app.py
```

The server will start at `http://0.0.0.0:7860`. Access it via:
- Local: `http://localhost:7860`
- Network: `http://<your-server-ip>:7860`

### Using the Web Interface

1. **Upload Audio**: Click the upload area or drag a `.wav` file
2. **Select Model**: Choose based on your accuracy/speed needs:
   - `tiny`: Fastest, least accurate
   - `base`: Fast, basic accuracy  
   - `small`: Good balance
   - `medium`: Recommended for most uses
   - `large-v3`: Best accuracy, slowest on CPU
3. **HuggingFace Token**: If you configured `.env`, you'll see a green checkmark. Otherwise, paste your token here.
4. **Set Speaker Limits** (optional): If you know the number of speakers
5. **Adjust CPU Threads**: Match your available cores (16-32 typical)
6. **Click Transcribe**: Wait for processing to complete
7. **Download**: Get the transcript as a `.txt` file

### Output Format

The transcript includes:
- File metadata (filename, model used, timestamp)
- Speaker-labeled segments with timestamps

Example:
```
Transcription of: meeting_recording.wav
Model: medium
Speaker diarization: Yes
Generated: 2024-01-15 14:30:00
============================================================

[00:00:05] SPEAKER_00: Welcome everyone to today's meeting. Let's start with the quarterly review.

[00:00:15] SPEAKER_01: Thanks for having us. I've prepared some slides on the sales figures.

[00:01:02] SPEAKER_00: Great, please go ahead and share your screen.
```

## Performance Expectations

Processing times on CPU (approximate, varies by hardware):

| Audio Length | Model Size | Est. Time (16 cores) |
|--------------|------------|----------------------|
| 30 min       | small      | 10-15 min            |
| 30 min       | medium     | 15-25 min            |
| 1 hour       | small      | 20-30 min            |
| 1 hour       | medium     | 30-50 min            |
| 3 hours      | small      | 60-90 min            |
| 3 hours      | medium     | 90-150 min           |
| 3 hours      | large-v3   | 3-6 hours            |

**Tips for faster processing:**
- Use more CPU threads (up to your core count)
- Use `small` model for drafts, `medium` or `large-v3` for final transcripts
- Ensure adequate RAM (processing loads full audio into memory)

## Running as a Service (Optional)

To run the transcription server as a systemd service:

```bash
sudo nano /etc/systemd/system/whisper-transcriber.service
```

Add:
```ini
[Unit]
Description=WhisperX Transcription Service
After=network.target

[Service]
Type=simple
User=your-username
WorkingDirectory=/path/to/whisper-transcriber
ExecStart=/path/to/whisper-transcriber/.venv/bin/python app.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable whisper-transcriber
sudo systemctl start whisper-transcriber
```

## Running Long Jobs

For multi-hour transcriptions, use `screen` or `tmux` to prevent interruption:

```bash
# Start a screen session
screen -S transcriber

# Run the app
uv run python app.py

# Detach with Ctrl+A, D
# Reattach later with: screen -r transcriber
```

## Troubleshooting

### "No module named 'whisperx'"
- Ensure you're using `uv run python`
- Reinstall dependencies: `uv sync`

### Diarization fails with authentication error
- Verify HuggingFace token is correct
- Ensure you've accepted licenses for both pyannote models
- Check token has read permissions

### Out of memory errors
- Use a smaller model (`small` instead of `medium`)
- Reduce batch size in the code
- Ensure no other memory-intensive processes are running

### Very slow processing
- Increase thread count in the UI
- Use a smaller model
- Check CPU usage - ensure all cores are being utilized

### FFmpeg errors
- Install FFmpeg: `sudo apt install ffmpeg`
- Verify: `ffmpeg -version`

### uv command not found
- Run: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Then: `source ~/.bashrc` or restart your shell

## File Structure

```
whisper-transcriber/
├── app.py              # Main application
├── pyproject.toml      # Project config and dependencies
├── uv.lock             # Locked dependency versions (generated)
├── setup.sh            # Automated setup script
├── .env.example        # Template for environment variables
├── .env                # Your local config (create from .env.example)
└── .venv/              # Virtual environment (created during setup)
```

## Why uv?

This project uses [uv](https://github.com/astral-sh/uv) for dependency management:
- **10-100x faster** package installation
- **Automatic Python version management** (downloads Python 3.11 if needed)
- **Lockfile support** for reproducible builds (`uv.lock`)
- **Single command setup** with `uv sync`

## License

This project is licensed under the [GNU General Public License v3.0](LICENSE).

Dependencies used:
- [WhisperX](https://github.com/m-bain/whisperx) - BSD-4-Clause
- [Gradio](https://github.com/gradio-app/gradio) - Apache-2.0
- [pyannote-audio](https://github.com/pyannote/pyannote-audio) - MIT (models require license acceptance)

## Acknowledgments

- OpenAI for the original Whisper model
- Max Bain for WhisperX
- pyannote team for speaker diarization
- Gradio team for the web framework
- Astral for uv
