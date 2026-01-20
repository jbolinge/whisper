"""
WhisperX Audio Transcription with Speaker Diarization
A Gradio-based web UI for transcribing audio files with speaker identification.
Optimized for CPU-only environments.
"""

import os
import tempfile
import time
from pathlib import Path
from typing import Optional
import gradio as gr
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Fix PyTorch 2.6+ compatibility: torch.load() now defaults to weights_only=True,
# which breaks loading WhisperX/pyannote models that contain custom serialized objects.
# This env var restores the previous behavior. See: https://github.com/m-bain/whisperX/issues/1304
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

# These imports will be available after installing requirements
import whisperx
import torch


def get_hf_token_from_env() -> Optional[str]:
    """Get HuggingFace token from environment variable."""
    return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")


def get_device_and_compute_type():
    """Determine the best device and compute type for the current hardware."""
    if torch.cuda.is_available():
        return "cuda", "float16"
    else:
        return "cpu", "int8"


def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_transcript_with_speakers(segments: list) -> str:
    """
    Format transcription segments into readable text with speaker labels and timestamps.
    
    Groups consecutive segments by speaker to avoid repetitive labels.
    """
    if not segments:
        return "No transcription results."
    
    lines = []
    current_speaker = None
    current_text_parts = []
    segment_start_time = None
    
    for segment in segments:
        speaker = segment.get("speaker", "UNKNOWN")
        text = segment.get("text", "").strip()
        start_time = segment.get("start", 0)
        
        if not text:
            continue
        
        # If speaker changed, write out the previous speaker's text
        if speaker != current_speaker:
            if current_speaker is not None and current_text_parts:
                timestamp = format_timestamp(segment_start_time)
                combined_text = " ".join(current_text_parts)
                lines.append(f"[{timestamp}] {current_speaker}: {combined_text}")
            
            # Start new speaker block
            current_speaker = speaker
            current_text_parts = [text]
            segment_start_time = start_time
        else:
            # Same speaker, accumulate text
            current_text_parts.append(text)
    
    # Don't forget the last speaker's text
    if current_speaker is not None and current_text_parts:
        timestamp = format_timestamp(segment_start_time)
        combined_text = " ".join(current_text_parts)
        lines.append(f"[{timestamp}] {current_speaker}: {combined_text}")
    
    return "\n\n".join(lines)


def format_transcript_simple(segments: list) -> str:
    """
    Format transcription without speaker labels (fallback when diarization fails).
    """
    if not segments:
        return "No transcription results."
    
    lines = []
    for segment in segments:
        text = segment.get("text", "").strip()
        start_time = segment.get("start", 0)
        
        if text:
            timestamp = format_timestamp(start_time)
            lines.append(f"[{timestamp}] {text}")
    
    return "\n\n".join(lines)


def transcribe_audio(
    audio_file: str,
    model_size: str = "medium",
    hf_token: Optional[str] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    num_threads: int = 8,
    progress: gr.Progress = gr.Progress()
) -> tuple[str, Optional[str]]:
    """
    Transcribe audio file with speaker diarization.
    
    Args:
        audio_file: Path to the audio file
        model_size: Whisper model size (tiny, base, small, medium, large-v3)
        hf_token: HuggingFace token for pyannote models (required for diarization)
        min_speakers: Minimum number of speakers (optional)
        max_speakers: Maximum number of speakers (optional)
        num_threads: Number of CPU threads to use
        progress: Gradio progress tracker
    
    Returns:
        Tuple of (transcript text, output file path)
    """
    if not audio_file:
        return "Please upload an audio file.", None
    
    # Check for token: UI input takes precedence, then fall back to env
    env_token = get_hf_token_from_env()
    effective_token = hf_token.strip() if hf_token and hf_token.strip() else env_token
    
    # Set thread count for CPU inference
    # OMP_NUM_THREADS controls CTranslate2 (used by faster-whisper/WhisperX)
    # torch.set_num_threads controls PyTorch operations (alignment, diarization)
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    torch.set_num_threads(num_threads)
    
    device, compute_type = get_device_and_compute_type()
    
    progress(0.05, desc="Loading Whisper model...")
    
    try:
        # Load WhisperX model
        model = whisperx.load_model(
            model_size,
            device=device,
            compute_type=compute_type,
            language="en"
        )
    except Exception as e:
        return f"Error loading model: {str(e)}", None
    
    progress(0.15, desc="Transcribing audio (this may take a while for long files)...")
    
    try:
        # Load and transcribe audio
        audio = whisperx.load_audio(audio_file)
        result = model.transcribe(audio, batch_size=16 if device == "cuda" else 4)
    except Exception as e:
        return f"Error during transcription: {str(e)}", None
    
    progress(0.50, desc="Aligning transcript...")
    
    try:
        # Align whisper output for better word-level timestamps
        model_a, metadata = whisperx.load_align_model(
            language_code="en",
            device=device
        )
        result = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            device,
            return_char_alignments=False
        )
        # Free up memory
        del model_a
    except Exception as e:
        progress(0.50, desc=f"Alignment warning: {str(e)}, continuing...")
    
    # Free whisper model memory
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    
    # Attempt speaker diarization if token available (from UI or env)
    diarization_success = False
    token_source = None
    
    if effective_token:
        if hf_token and hf_token.strip():
            token_source = "UI input"
        else:
            token_source = ".env file"
        
        progress(0.65, desc=f"Performing speaker diarization (token from {token_source})...")
        
        try:
            diarize_model = whisperx.DiarizationPipeline(
                use_auth_token=effective_token,
                device=device
            )
            
            diarize_kwargs = {}
            if min_speakers and min_speakers > 0:
                diarize_kwargs["min_speakers"] = min_speakers
            if max_speakers and max_speakers > 0:
                diarize_kwargs["max_speakers"] = max_speakers
            
            diarize_segments = diarize_model(audio, **diarize_kwargs)
            result = whisperx.assign_word_speakers(diarize_segments, result)
            diarization_success = True
            
            del diarize_model
            if device == "cuda":
                torch.cuda.empty_cache()
                
        except Exception as e:
            progress(0.65, desc=f"Diarization failed: {str(e)}, continuing without speaker labels...")
    
    progress(0.90, desc="Formatting output...")
    
    # Format the transcript
    segments = result.get("segments", [])
    
    if diarization_success:
        transcript = format_transcript_with_speakers(segments)
    else:
        transcript = format_transcript_simple(segments)
        if not effective_token:
            transcript = "NOTE: No HuggingFace token provided (neither in UI nor .env file) - speaker diarization disabled.\n\n" + transcript
    
    # Save to file
    progress(0.95, desc="Saving transcript...")
    
    output_dir = tempfile.mkdtemp()
    input_filename = Path(audio_file).stem
    output_path = os.path.join(output_dir, f"{input_filename}_transcript.txt")
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"Transcription of: {Path(audio_file).name}\n")
        f.write(f"Model: {model_size}\n")
        f.write(f"Speaker diarization: {'Yes' if diarization_success else 'No'}\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        f.write(transcript)
    
    progress(1.0, desc="Complete!")
    
    return transcript, output_path


def create_ui():
    """Create and configure the Gradio interface."""
    
    # Check if token is available from environment
    env_token = get_hf_token_from_env()
    token_from_env = env_token is not None and len(env_token) > 0
    
    with gr.Blocks(
        title="Audio Transcription with Speaker Diarization"
    ) as app:
        gr.Markdown(
            """
            # 🎙️ Audio Transcription with Speaker Diarization
            
            Upload a `.wav` audio file to transcribe it with automatic speaker identification.
            
            **Features:**
            - Transcription using WhisperX (optimized for long-form audio)
            - Speaker diarization (who said what)
            - Timestamps for each speaker segment
            - Plain text output
            
            **Note:** For speaker diarization, you need a HuggingFace token with access to 
            [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1).
            Accept the license there first, then either add it to your `.env` file or paste it below.
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    label="Upload Audio File",
                    type="filepath",
                    sources=["upload"]
                )
                
                model_dropdown = gr.Dropdown(
                    choices=["tiny", "base", "small", "medium", "large-v3"],
                    value="medium",
                    label="Model Size",
                    info="Larger models are more accurate but slower. 'medium' recommended for CPU."
                )
                
                # Show different UI based on whether token is in .env
                if token_from_env:
                    gr.Markdown(
                        """
                        ✅ **HuggingFace token loaded from `.env` file**
                        
                        Speaker diarization is enabled. You can override the token below if needed.
                        """
                    )
                    hf_token = gr.Textbox(
                        label="HuggingFace Token (optional - override .env)",
                        placeholder="Leave empty to use token from .env",
                        type="password",
                        info="Token already loaded from .env file"
                    )
                else:
                    hf_token = gr.Textbox(
                        label="HuggingFace Token (for speaker diarization)",
                        placeholder="hf_...",
                        type="password",
                        info="Required for speaker identification. Get one at huggingface.co/settings/tokens or add HF_TOKEN to .env file"
                    )
                
                with gr.Row():
                    min_speakers = gr.Number(
                        label="Min Speakers",
                        value=None,
                        precision=0,
                        minimum=1,
                        maximum=20,
                        info="Optional: minimum expected speakers"
                    )
                    max_speakers = gr.Number(
                        label="Max Speakers",
                        value=None,
                        precision=0,
                        minimum=1,
                        maximum=20,
                        info="Optional: maximum expected speakers"
                    )
                
                num_threads = gr.Slider(
                    minimum=1,
                    maximum=64,
                    value=16,
                    step=1,
                    label="CPU Threads",
                    info="Number of CPU threads for processing"
                )
                
                transcribe_btn = gr.Button(
                    "🚀 Transcribe",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=2):
                output_text = gr.Textbox(
                    label="Transcript",
                    lines=25,
                    max_lines=50
                )
                
                output_file = gr.File(
                    label="Download Transcript"
                )
        
        gr.Markdown(
            """
            ---
            ### Tips for Best Results:
            - **Audio Quality:** Clear audio with minimal background noise works best
            - **Model Selection:** 
              - `tiny`/`base`: Fast but less accurate
              - `small`/`medium`: Good balance (recommended for CPU)
              - `large-v3`: Most accurate but slow on CPU
            - **Speaker Diarization:** Works best when speakers have distinct voices and don't talk over each other
            - **Long Files:** 3-hour files may take 30-60+ minutes on CPU depending on model size
            """
        )
        
        # Connect the transcribe button
        transcribe_btn.click(
            fn=transcribe_audio,
            inputs=[
                audio_input,
                model_dropdown,
                hf_token,
                min_speakers,
                max_speakers,
                num_threads
            ],
            outputs=[output_text, output_file]
        )
    
    return app


if __name__ == "__main__":
    app = create_ui()
    app.launch(
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860,
        share=False,  # Set to True if you want a public URL
        theme=gr.themes.Soft()
    )
