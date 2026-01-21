"""Nemotron ASR model wrapper for streaming transcription."""

import logging
import tempfile
import os
import threading
import time
from dataclasses import dataclass
from typing import Iterator
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionResult:
    """Result from transcription."""
    text: str
    is_final: bool
    confidence: float = 1.0


class NemotronTranscriber:
    """Wrapper for Nemotron ASR model.

    Uses NVIDIA NeMo for transcription. Accumulates audio in a buffer
    and transcribes periodically.
    """

    MODEL_NAME = "nvidia/nemotron-speech-streaming-en-0.6b"
    SAMPLE_RATE = 16000  # Required: 16kHz mono
    # RMS energy threshold for voice activity detection (normalized float audio)
    # Values below this are considered silence/noise
    # 0.01 = very quiet, 0.05 = quiet speech, 0.1 = normal speech
    SILENCE_THRESHOLD = 0.005  # Very low - catches most silence

    def __init__(
        self,
        model_name: str | None = None,
        buffer_duration_s: float = 0.5,  # Min duration to transcribe (Rust does VAD now)
        silence_threshold: float | None = None,
    ):
        """Initialize the transcriber.

        Args:
            model_name: HuggingFace model name. Defaults to Nemotron.
            buffer_duration_s: Minimum audio duration before transcribing.
            silence_threshold: RMS threshold below which audio is considered silent.
        """
        self.model_name = model_name or self.MODEL_NAME
        self.buffer_duration_s = buffer_duration_s
        self.silence_threshold = silence_threshold or self.SILENCE_THRESHOLD
        self._model = None
        self._audio_buffer: list[np.ndarray] = []
        self._last_transcription = ""
        self._recent_transcriptions: list[str] = []  # For deduplication
        self._lock = threading.Lock()  # Thread safety for buffer operations
        self._last_transcribe_time = 0.0
        self._chunk_count = 0

    def load_model(self) -> None:
        """Load the Nemotron model. Downloads if not cached."""
        try:
            import nemo.collections.asr as nemo_asr
        except ImportError as e:
            raise ImportError(
                "NeMo toolkit not installed. Install with: "
                "pip install 'nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git@main'"
            ) from e

        logger.info(f"Loading model: {self.model_name}")
        self._model = nemo_asr.models.ASRModel.from_pretrained(
            model_name=self.model_name
        )
        logger.info("Model loaded successfully")

    def start_session(self) -> None:
        """Start a new transcription session."""
        if self._model is None:
            self.load_model()
        with self._lock:
            self._audio_buffer = []
            self._last_transcription = ""
            self._recent_transcriptions = []  # Clear dedup history
            self._chunk_count = 0
            self._last_transcribe_time = 0.0
        logger.info("Started new transcription session")

    def end_session(self) -> None:
        """End the current transcription session."""
        with self._lock:
            self._audio_buffer = []
            self._last_transcription = ""
            self._chunk_count = 0
        logger.debug("Ended transcription session")

    def add_audio(self, audio_chunk: np.ndarray) -> TranscriptionResult | None:
        """Add audio chunk to buffer and transcribe if buffer is full.

        Args:
            audio_chunk: Audio samples as int16 or float32 numpy array.
                        Expected shape: (num_samples,) at 16kHz.

        Returns:
            TranscriptionResult if transcription was triggered, None otherwise.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Convert to float32 and normalize
        if audio_chunk.dtype == np.int16:
            audio_chunk = audio_chunk.astype(np.float32) / 32768.0
        elif audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)

        # Thread-safe buffer operations
        with self._lock:
            self._chunk_count += 1
            self._audio_buffer.append(audio_chunk)

            # Check if we have enough audio
            total_samples = sum(len(c) for c in self._audio_buffer)
            buffer_duration = total_samples / self.SAMPLE_RATE

            # Log every 30 chunks (~3s worth at 100ms chunks)
            if self._chunk_count % 30 == 0:
                logger.info(f"Audio buffer: {total_samples} samples ({buffer_duration:.2f}s), "
                           f"chunks: {len(self._audio_buffer)}, chunk_size: {len(audio_chunk)}")

            if buffer_duration >= self.buffer_duration_s:
                # Calculate timing since last transcription
                now = time.time()
                time_since_last = now - self._last_transcribe_time if self._last_transcribe_time > 0 else 0
                self._last_transcribe_time = now

                # Calculate RMS before transcribing
                audio = np.concatenate(self._audio_buffer)
                rms = self._calculate_rms(audio)
                logger.info(f">>> Transcribing: {buffer_duration:.2f}s audio, RMS={rms:.4f}, "
                           f"time_since_last={time_since_last:.1f}s")

                # Clear buffer BEFORE releasing lock to prevent race conditions
                audio_to_transcribe = audio
                self._audio_buffer = []

        # Transcribe outside the lock (this is slow)
        if buffer_duration >= self.buffer_duration_s:
            result = self._transcribe_audio(audio_to_transcribe)
            if result.text:
                logger.info(f">>> Result: '{result.text}'")
            return result

        return None

    def _calculate_rms(self, audio: np.ndarray) -> float:
        """Calculate RMS (root mean square) energy of audio."""
        return float(np.sqrt(np.mean(audio ** 2)))

    def _transcribe_audio(self, audio: np.ndarray) -> TranscriptionResult:
        """Transcribe pre-extracted audio data.

        Args:
            audio: Float32 normalized audio at 16kHz.

        Returns:
            TranscriptionResult with transcribed text.
        """
        # Check if audio has sufficient energy (voice activity detection)
        rms = self._calculate_rms(audio)
        if rms < self.silence_threshold:
            logger.debug(f"Skipping silent audio (RMS={rms:.4f} < threshold={self.silence_threshold})")
            return TranscriptionResult(text="", is_final=False)

        logger.info(f"Transcribing {len(audio)/self.SAMPLE_RATE:.1f}s audio, RMS={rms:.4f}")

        # Save to temp file for transcription (NeMo's transcribe expects files)
        import scipy.io.wavfile as wav
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
            # Convert to int16 for wav file
            audio_int16 = (audio * 32767).astype(np.int16)
            wav.write(temp_path, self.SAMPLE_RATE, audio_int16)

        try:
            # Transcribe (NeMo outputs progress bars, which is the "Transcribing:" output)
            results = self._model.transcribe([temp_path])
            text = results[0] if results else ""

            # Extract text if it's a Hypothesis object
            if hasattr(text, 'text'):
                text = text.text

            # Strip whitespace and skip if empty
            text = text.strip() if text else ""

            # Skip common hallucination patterns
            if self._is_likely_hallucination(text):
                logger.info(f"Filtered likely hallucination: '{text}'")
                return TranscriptionResult(text="", is_final=False)

            # Check for duplicate/similar transcriptions
            if self._is_duplicate(text):
                logger.info(f"Filtered duplicate: '{text}'")
                return TranscriptionResult(text="", is_final=False)

            # Track recent transcriptions for deduplication
            self._recent_transcriptions.append(text.lower())
            if len(self._recent_transcriptions) > 5:
                self._recent_transcriptions.pop(0)

            self._last_transcription = text
            logger.info(f">>> Transcription: '{text}'")
            return TranscriptionResult(text=text, is_final=False)
        finally:
            os.unlink(temp_path)

    def _is_likely_hallucination(self, text: str) -> bool:
        """Check if text is likely a hallucination."""
        if not text:
            return False

        text_lower = text.lower().strip()

        # Common hallucination patterns
        hallucination_patterns = [
            "thank you",
            "thanks for watching",
            "subscribe",
            "like and subscribe",
            "see you next time",
            "bye bye",
            "goodbye",
            "music",
            "applause",
            "laughter",
            "[music]",
            "[applause]",
        ]

        for pattern in hallucination_patterns:
            if text_lower == pattern or text_lower.startswith(pattern + "."):
                return True

        # Very short single words are often hallucinations
        if len(text_lower) < 3 and " " not in text_lower:
            return True

        # Repeated characters/words often indicate hallucination
        words = text_lower.split()
        if len(words) >= 3 and len(set(words)) == 1:
            return True  # Same word repeated 3+ times

        return False

    def _is_duplicate(self, text: str) -> bool:
        """Check if text is a duplicate of recent transcriptions."""
        if not text or not self._recent_transcriptions:
            return False

        text_lower = text.lower().strip()

        for recent in self._recent_transcriptions:
            # Exact match
            if text_lower == recent:
                return True
            # High similarity (one is substring of other)
            if len(text_lower) > 10 and len(recent) > 10:
                if text_lower in recent or recent in text_lower:
                    return True

        return False

    def _transcribe_buffer(self) -> TranscriptionResult:
        """Transcribe the accumulated audio buffer (thread-safe)."""
        with self._lock:
            if not self._audio_buffer:
                return TranscriptionResult(text="", is_final=False)

            # Concatenate all audio chunks and clear buffer
            audio = np.concatenate(self._audio_buffer)
            self._audio_buffer = []

        # Transcribe outside the lock
        return self._transcribe_audio(audio)

    def flush(self) -> TranscriptionResult:
        """Transcribe any remaining audio in the buffer (thread-safe)."""
        with self._lock:
            if not self._audio_buffer:
                return TranscriptionResult(text="", is_final=True)
            audio = np.concatenate(self._audio_buffer)
            self._audio_buffer = []

        result = self._transcribe_audio(audio)
        return TranscriptionResult(text=result.text, is_final=True, confidence=result.confidence)

    def transcribe_file(self, audio_path: str) -> str:
        """Transcribe an entire audio file.

        Args:
            audio_path: Path to audio file (wav, mp3, etc.)

        Returns:
            Full transcription text.
        """
        if self._model is None:
            self.load_model()

        results = self._model.transcribe([audio_path])
        text = results[0] if results else ""

        # Extract text if it's a Hypothesis object
        if hasattr(text, 'text'):
            text = text.text

        return text

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None

    @property
    def buffer_samples(self) -> int:
        """Target number of samples for buffer."""
        return int(self.SAMPLE_RATE * self.buffer_duration_s)
