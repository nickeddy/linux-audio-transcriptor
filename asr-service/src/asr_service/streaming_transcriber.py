"""Streaming ASR transcriber using cache-aware inference."""

import logging
import threading
import numpy as np
import torch
from dataclasses import dataclass
from typing import Optional
import tempfile
import os

logger = logging.getLogger(__name__)


@dataclass
class StreamingResult:
    """Result from streaming transcription."""
    text: str
    is_partial: bool  # True for partial, False for final
    segment_id: int   # Increments for each new utterance
    speaker: str = "Speaker"  # Speaker label from diarization


class StreamingTranscriber:
    """Cache-aware streaming transcriber using Nemotron.

    Uses NeMo's CacheAwareStreamingAudioBuffer for proper
    preprocessing and cache management.
    """

    MODEL_NAME = "nvidia/nemotron-speech-streaming-en-0.6b"
    SAMPLE_RATE = 16000

    def __init__(
        self,
        model_name: str | None = None,
        chunk_size_ms: int = 160,
    ):
        """Initialize the streaming transcriber.

        Args:
            model_name: HuggingFace model name. Defaults to Nemotron.
            chunk_size_ms: Chunk size in ms (80, 160, 560, or 1120).
        """
        self.model_name = model_name or self.MODEL_NAME
        self.chunk_size_ms = chunk_size_ms

        self._model = None
        self._device = None
        self._dtype = None

        # Streaming components
        self._streaming_buffer = None
        self._cache_last_channel = None
        self._cache_last_time = None
        self._cache_last_channel_len = None
        self._previous_hypotheses = None
        self._pred_out_stream = None
        self._step_num = 0

        # Audio accumulation buffer (raw audio before adding to streaming buffer)
        self._audio_buffer = np.array([], dtype=np.float32)
        self._samples_per_chunk = int(self.SAMPLE_RATE * chunk_size_ms / 1000)

        # Segment tracking
        self._segment_id = 0
        self._last_text = ""

        # Speaker diarization
        self._speaker_model = None
        self._speaker_embeddings: dict[str, np.ndarray] = {}  # speaker_id -> embedding
        self._current_speaker = "Speaker 1"
        self._speaker_count = 1
        self._segment_audio = np.array([], dtype=np.float32)  # Audio for current segment
        self._similarity_threshold = 0.5  # Cosine similarity threshold for same speaker

        # Thread safety
        self._lock = threading.Lock()

    def load_model(self) -> None:
        """Load the Nemotron model for streaming inference."""
        try:
            import nemo.collections.asr as nemo_asr
            from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer
        except ImportError as e:
            raise ImportError(
                "NeMo toolkit not installed. Install with: "
                "pip install 'nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git@main'"
            ) from e

        logger.info(f"Loading streaming model: {self.model_name}")

        self._model = nemo_asr.models.ASRModel.from_pretrained(
            model_name=self.model_name
        )

        # Set up device and dtype
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self._model = self._model.to(device=self._device, dtype=self._dtype)
        self._model.eval()

        # Configure streaming parameters
        # chunk_size is in frames (after preprocessing), not raw samples
        # For Conformer with 4x subsampling and 10ms frame shift: 160ms = 16 frames
        chunk_frames = self.chunk_size_ms // 10  # Approximate frame count

        if hasattr(self._model.encoder, 'setup_streaming_params'):
            self._model.encoder.setup_streaming_params(
                chunk_size=chunk_frames,
                left_chunks=4,  # Context chunks
                shift_size=chunk_frames,
            )
            logger.info(f"Configured streaming: chunk_frames={chunk_frames}")

        # Store the buffer class for later instantiation
        self._CacheAwareStreamingAudioBuffer = CacheAwareStreamingAudioBuffer

        # Load TitaNet for speaker embeddings
        logger.info("Loading TitaNet speaker model...")
        try:
            self._speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
                model_name="nvidia/speakerverification_en_titanet_large"
            )
            self._speaker_model = self._speaker_model.to(device=self._device)
            self._speaker_model.eval()
            logger.info("TitaNet speaker model loaded successfully")
        except Exception as e:
            logger.error(f"Could not load TitaNet speaker model: {e}")
            import traceback
            traceback.print_exc()
            self._speaker_model = None

        logger.info(f"Model loaded on {self._device}")

    def start_session(self) -> None:
        """Start a new streaming session - reset all state."""
        if self._model is None:
            self.load_model()

        with self._lock:
            # Create fresh streaming buffer
            self._streaming_buffer = self._CacheAwareStreamingAudioBuffer(
                model=self._model,
                online_normalization=True,
                pad_and_drop_preencoded=False,
            )

            # Initialize cache state and convert to correct device/dtype
            cache = self._model.encoder.get_initial_cache_state(batch_size=1)
            self._cache_last_channel = cache[0].to(device=self._device, dtype=self._dtype)
            self._cache_last_time = cache[1].to(device=self._device, dtype=self._dtype)
            self._cache_last_channel_len = cache[2] if len(cache) > 2 else None

            self._previous_hypotheses = None
            self._pred_out_stream = None
            self._step_num = 0

            self._audio_buffer = np.array([], dtype=np.float32)
            self._segment_audio = np.array([], dtype=np.float32)
            self._segment_id = 0
            self._last_text = ""

            # Reset speaker tracking for new session
            self._speaker_embeddings = {}
            self._current_speaker = "Speaker 1"
            self._speaker_count = 1

        logger.info("Started new streaming session")

    def end_session(self) -> Optional[StreamingResult]:
        """End the session and return any final transcription."""
        with self._lock:
            result = None
            if self._last_text:
                result = StreamingResult(
                    text=self._last_text,
                    is_partial=False,
                    segment_id=self._segment_id,
                )

            self._streaming_buffer = None
            self._audio_buffer = np.array([], dtype=np.float32)

        logger.info("Ended streaming session")
        return result

    def add_audio(self, audio_chunk: np.ndarray) -> Optional[StreamingResult]:
        """Add audio chunk and get streaming transcription.

        Args:
            audio_chunk: Audio samples as int16 or float32 numpy array.
                        Expected shape: (num_samples,) at 16kHz.

        Returns:
            StreamingResult if transcription is available, None otherwise.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Convert to float32 and normalize
        if audio_chunk.dtype == np.int16:
            audio_chunk = audio_chunk.astype(np.float32) / 32768.0
        elif audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)

        with self._lock:
            # Accumulate audio
            self._audio_buffer = np.concatenate([self._audio_buffer, audio_chunk])

            # Process when we have enough for a chunk
            result = None
            while len(self._audio_buffer) >= self._samples_per_chunk:
                chunk = self._audio_buffer[:self._samples_per_chunk]
                self._audio_buffer = self._audio_buffer[self._samples_per_chunk:]

                chunk_result = self._process_chunk(chunk)
                if chunk_result:
                    result = chunk_result

            return result

    def _process_chunk(self, chunk: np.ndarray) -> Optional[StreamingResult]:
        """Process a single audio chunk through cache-aware streaming."""
        if self._streaming_buffer is None:
            logger.warning("No streaming buffer - session not started")
            return None

        try:
            # Add audio to the streaming buffer
            # append_audio expects a numpy array, not a tensor
            # Accumulate audio for speaker identification
            self._segment_audio = np.concatenate([self._segment_audio, chunk])

            # append_audio returns (processed_signal, processed_signal_length, stream_id)
            processed_signal, processed_length, _ = self._streaming_buffer.append_audio(
                audio=chunk,
            )

            # Check if we got processed output
            if processed_signal is None or processed_signal.size(0) == 0:
                return None

            processed_signal = processed_signal.to(device=self._device, dtype=self._dtype)

            # Ensure processed_length has batch dimension (batch,)
            if processed_length.dim() == 0:
                processed_length = processed_length.unsqueeze(0)
            processed_length = processed_length.to(device=self._device)

            # Calculate drop_extra_pre_encoded: 0 on first step, otherwise use model's config
            if self._step_num == 0:
                drop_extra = 0
            else:
                drop_extra = getattr(
                    self._model.encoder.streaming_cfg, 'drop_extra_pre_encoded', 0
                )

            with torch.no_grad():
                (
                    self._pred_out_stream,
                    transcribed_texts,
                    self._cache_last_channel,
                    self._cache_last_time,
                    self._cache_last_channel_len,
                    self._previous_hypotheses,
                ) = self._model.conformer_stream_step(
                    processed_signal=processed_signal,
                    processed_signal_length=processed_length,
                    cache_last_channel=self._cache_last_channel,
                    cache_last_time=self._cache_last_time,
                    cache_last_channel_len=self._cache_last_channel_len,
                    keep_all_outputs=self._streaming_buffer.is_buffer_empty(),
                    previous_hypotheses=self._previous_hypotheses,
                    previous_pred_out=self._pred_out_stream,
                    drop_extra_pre_encoded=drop_extra,
                    return_transcription=True,
                )

            self._step_num += 1

            # Extract text from results
            if transcribed_texts:
                text = transcribed_texts[0]
                if hasattr(text, 'text'):
                    text = text.text
                text = text.strip() if text else ""

                if text and text != self._last_text:
                    self._last_text = text
                    return StreamingResult(
                        text=text,
                        is_partial=True,
                        segment_id=self._segment_id,
                    )

            return None

        except Exception as e:
            logger.error(f"Error processing chunk: {e}")
            import traceback
            traceback.print_exc()
            return None

    def finalize_segment(self) -> Optional[StreamingResult]:
        """Mark current segment as final and start a new one.

        Call this when silence is detected to finalize the current
        utterance and prepare for the next one.
        """
        logger.info(f"finalize_segment called, last_text='{self._last_text[:50] if self._last_text else None}...', segment_audio_len={len(self._segment_audio)}")

        # First, grab what we need under the lock
        with self._lock:
            if not self._last_text:
                logger.info("No text to finalize, returning None")
                return None

            text = self._last_text
            segment_id = self._segment_id
            segment_audio = self._segment_audio.copy() if len(self._segment_audio) > 0 else None
            current_speaker = self._current_speaker

            # Reset for next segment (do this now to unblock audio processing)
            self._segment_id += 1
            self._last_text = ""
            self._audio_buffer = np.array([], dtype=np.float32)
            self._segment_audio = np.array([], dtype=np.float32)

            # Reset streaming state for new utterance
            if self._streaming_buffer is not None:
                self._streaming_buffer = self._CacheAwareStreamingAudioBuffer(
                    model=self._model,
                    online_normalization=True,
                    pad_and_drop_preencoded=False,
                )
                cache = self._model.encoder.get_initial_cache_state(batch_size=1)
                self._cache_last_channel = cache[0].to(device=self._device, dtype=self._dtype)
                self._cache_last_time = cache[1].to(device=self._device, dtype=self._dtype)
                self._cache_last_channel_len = cache[2] if len(cache) > 2 else None
                self._previous_hypotheses = None
                self._pred_out_stream = None
                self._step_num = 0

        # Now do speaker identification OUTSIDE the lock (this can be slow)
        speaker = current_speaker
        if segment_audio is not None and len(segment_audio) > 0:
            segment_duration = len(segment_audio) / self.SAMPLE_RATE
            logger.info(f"Identifying speaker from {segment_duration:.2f}s audio")

            try:
                embedding = self._extract_speaker_embedding(segment_audio)
                if embedding is not None:
                    # Need lock for speaker tracking state
                    with self._lock:
                        speaker = self._identify_speaker(embedding)
            except Exception as e:
                logger.error(f"Speaker identification failed: {e}")

        result = StreamingResult(
            text=text,
            is_partial=False,
            segment_id=segment_id,
            speaker=speaker,
        )
        logger.info(f"Final [{segment_id}] {speaker}: {text}")
        return result

    def _extract_speaker_embedding(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """Extract speaker embedding from audio using TitaNet."""
        if self._speaker_model is None:
            logger.debug("Speaker model not loaded, skipping embedding")
            return None

        # Need at least 0.5 seconds of audio for reliable embedding
        if len(audio) < self.SAMPLE_RATE // 2:
            logger.debug(f"Audio too short for embedding: {len(audio)} samples")
            return None

        try:
            # Save to temp file (TitaNet expects file paths)
            import scipy.io.wavfile as wav
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = f.name
                audio_int16 = (audio * 32767).astype(np.int16)
                wav.write(temp_path, self.SAMPLE_RATE, audio_int16)

            try:
                # Extract embedding
                logger.debug(f"Extracting embedding from {temp_path}")
                with torch.no_grad():
                    embedding = self._speaker_model.get_embedding(temp_path)
                    result = embedding.cpu().numpy().flatten()
                    logger.debug(f"Embedding extracted: shape={result.shape}")
                    return result
            finally:
                os.unlink(temp_path)

        except Exception as e:
            logger.error(f"Failed to extract speaker embedding: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _identify_speaker(self, embedding: np.ndarray) -> str:
        """Identify speaker from embedding by comparing to known speakers."""
        if len(self._speaker_embeddings) == 0:
            # First speaker
            speaker = f"Speaker {self._speaker_count}"
            self._speaker_embeddings[speaker] = embedding
            self._current_speaker = speaker
            logger.info(f"New speaker detected: {speaker}")
            return speaker

        # Compare to all known speakers using cosine similarity
        best_match = None
        best_similarity = -1.0

        for speaker_id, known_embedding in self._speaker_embeddings.items():
            similarity = self._cosine_similarity(embedding, known_embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = speaker_id

        if best_similarity >= self._similarity_threshold:
            # Match found - update embedding with running average
            old_emb = self._speaker_embeddings[best_match]
            self._speaker_embeddings[best_match] = 0.7 * old_emb + 0.3 * embedding
            self._current_speaker = best_match
            logger.debug(f"Speaker matched: {best_match} (similarity: {best_similarity:.3f})")
            return best_match
        else:
            # New speaker
            self._speaker_count += 1
            speaker = f"Speaker {self._speaker_count}"
            self._speaker_embeddings[speaker] = embedding
            self._current_speaker = speaker
            logger.info(f"New speaker detected: {speaker} (best similarity was {best_similarity:.3f})")
            return speaker

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None
