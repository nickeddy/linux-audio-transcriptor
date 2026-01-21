#!/usr/bin/env python3
"""Test microphone recording and ASR transcription."""

import sys
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
import tempfile
import os

SAMPLE_RATE = 16000  # Nemotron requires 16kHz


def list_devices():
    """List available audio devices."""
    print("\nAvailable audio devices:")
    print(sd.query_devices())
    print(f"\nDefault input device: {sd.default.device[0]}")


def record_audio(duration: float = 5.0) -> np.ndarray:
    """Record audio from microphone.

    Args:
        duration: Recording duration in seconds.

    Returns:
        Audio samples as int16 numpy array.
    """
    print(f"\n🎤 Recording for {duration} seconds... Speak now!")

    audio = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype=np.int16,
    )
    sd.wait()  # Wait for recording to finish

    print("✓ Recording complete!")
    return audio.flatten()


def transcribe(audio: np.ndarray) -> str:
    """Transcribe audio using Nemotron."""
    import nemo.collections.asr as nemo_asr

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_path = f.name
        wav.write(temp_path, SAMPLE_RATE, audio)

    try:
        print("\n🔄 Loading model (first run downloads ~1GB)...")
        model = nemo_asr.models.ASRModel.from_pretrained(
            "nvidia/nemotron-speech-streaming-en-0.6b"
        )

        print("🔄 Transcribing...")
        results = model.transcribe([temp_path])

        text = results[0] if results else ""
        # Extract text if it's a Hypothesis object
        if hasattr(text, 'text'):
            text = text.text

        return text
    finally:
        os.unlink(temp_path)


def main():
    """Main entry point."""
    print("=" * 50)
    print("  Microphone ASR Test (Nemotron)")
    print("=" * 50)

    # Check for --list-devices flag
    if "--list-devices" in sys.argv:
        list_devices()
        return

    # Get recording duration
    duration = 5.0
    if len(sys.argv) > 1:
        try:
            duration = float(sys.argv[1])
        except ValueError:
            pass

    print(f"\nWill record for {duration} seconds.")
    print("Press Enter to start recording, or Ctrl+C to cancel...")

    try:
        input()
    except KeyboardInterrupt:
        print("\nCancelled.")
        return

    # Record
    audio = record_audio(duration)

    # Check if we got audio
    max_amplitude = np.abs(audio).max()
    print(f"Max amplitude: {max_amplitude} (should be > 100 if you spoke)")

    if max_amplitude < 100:
        print("⚠️  Very quiet recording. Check your microphone.")
        return

    # Transcribe
    text = transcribe(audio)

    print("\n" + "=" * 50)
    print("  TRANSCRIPTION RESULT")
    print("=" * 50)
    print(f"\n  \"{text}\"\n")
    print("=" * 50)


if __name__ == "__main__":
    main()
