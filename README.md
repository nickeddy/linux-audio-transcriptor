# Linux Audio Transcriptor

Live meeting transcription and summarization for Linux. Captures both microphone and system audio (remote meeting participants), transcribes in real-time using NVIDIA Nemotron, and generates summaries using LFM2.

## Features

- **Real-time streaming transcription** using NVIDIA Nemotron-Speech-Streaming (0.6B parameters)
- **Dual audio capture**: microphone (via ALSA/cpal) + system audio (via PipeWire)
- **Speaker diarization** using TitaNet embeddings
- **Meeting summarization** using LFM2-2.6B-Transcript (via llama.cpp)
- **Terminal UI** built with Ratatui
- **Export** transcripts to Markdown

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Rust TUI Application                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ Mic Capture │  │  PipeWire   │  │   Summarization     │  │
│  │   (cpal)    │  │   Capture   │  │  (llama-cpp/LFM2)   │  │
│  └──────┬──────┘  └──────┬──────┘  └─────────────────────┘  │
│         │                │                                   │
│         └───────┬────────┘                                   │
│                 │ 16kHz mono PCM                             │
│                 ▼                                            │
│         ┌──────────────┐                                     │
│         │  VAD + Send  │                                     │
│         └──────┬───────┘                                     │
│                │ WebSocket                                   │
└────────────────┼────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│                  Python ASR Service                          │
│  ┌─────────────────────────────────────────────────────┐    │
│  │           StreamingTranscriber (NeMo)               │    │
│  │  • CacheAwareStreamingAudioBuffer                   │    │
│  │  • conformer_stream_step (cache-aware inference)    │    │
│  │  • TitaNet speaker embeddings                       │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## Requirements

### System
- Linux with PipeWire (for system audio capture)
- NVIDIA GPU with CUDA (for ASR inference)
- Rust 1.70+
- Python 3.10+

### Models
- **ASR**: `nvidia/nemotron-speech-streaming-en-0.6b` (auto-downloaded)
- **Speaker ID**: `nvidia/speakerverification_en_titanet_large` (auto-downloaded)
- **Summarization**: LFM2-2.6B-Transcript-GGUF (manual download)

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/linux-audio-transcriptor.git
cd linux-audio-transcriptor
```

### 2. Set up the Python ASR service

```bash
cd asr-service
python -m venv .venv
source .venv/bin/activate
pip install -e .

# Install NeMo (requires CUDA)
pip install 'nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git@main'
pip install scipy  # For speaker diarization
```

### 3. Build the Rust application

```bash
cd ..
cargo build --release
```

### 4. (Optional) Download LFM2 for summarization

```bash
huggingface-cli download LiquidAI/LFM2-2.6B-Transcript-GGUF \
    --local-dir ~/.cache/lfm2/
```

## Usage

### 1. Start the ASR service

```bash
cd asr-service
source .venv/bin/activate
PYTHONPATH=src python -m asr_service.main

# With verbose logging:
PYTHONPATH=src python -m asr_service.main -v

# Custom chunk size (80, 160, 560, or 1120 ms):
PYTHONPATH=src python -m asr_service.main --chunk-size 160
```

### 2. Run the TUI application

```bash
# In another terminal
cargo run --release

# Or with options:
cargo run --release -- --asr-url ws://127.0.0.1:8765 --title "Team Standup"
```

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Space` | Start/stop recording |
| `s` | Generate summary |
| `e` | Export transcript to Markdown |
| `Tab` | Switch between Transcript/Summary panels |
| `Up/Down` | Scroll active panel |
| `r` | Reconnect to ASR server |
| `q` | Quit |

## Configuration

Configuration is stored in `~/.config/linux-audio-transcriptor/config.toml`:

```toml
[audio]
sample_rate = 16000
capture_mic = true
capture_system = true

[llm]
# Path to GGUF model file
model_path = "~/.cache/lfm2/lfm2-2.6b-transcript.Q4_K_M.gguf"
n_threads = 4
n_ctx = 4096
temperature = 0.7

[output]
directory = "~/transcripts"
```

## Project Structure

```
linux-audio-transcriptor/
├── src/
│   ├── main.rs              # Entry point
│   ├── config.rs            # Configuration management
│   ├── session.rs           # Meeting session state
│   ├── asr_client.rs        # WebSocket client for ASR
│   ├── audio/
│   │   ├── capture.rs       # Microphone capture (cpal)
│   │   └── pipewire_capture.rs  # System audio capture
│   ├── ui/
│   │   ├── app.rs           # Application state & main loop
│   │   └── views.rs         # TUI rendering
│   └── summarization/
│       └── llm.rs           # LFM2 summarization
├── asr-service/
│   └── src/asr_service/
│       ├── main.py          # ASR server entry point
│       ├── server.py        # WebSocket server
│       ├── streaming_transcriber.py  # Cache-aware streaming ASR
│       └── transcriber.py   # Batch transcriber (fallback)
├── Cargo.toml
└── README.md
```

## Troubleshooting

### No audio being captured
- Check PipeWire is running: `systemctl --user status pipewire`
- Verify microphone permissions
- Check audio devices: `cargo run --example list_devices`

### ASR connection failed
- Ensure the ASR service is running on the correct port
- Check firewall settings for localhost connections

### CUDA out of memory
- Try reducing chunk size: `--chunk-size 80`
- Ensure no other GPU processes are running

### Speaker diarization not working
- Check TitaNet model loaded successfully in ASR logs
- Segments need >0.5s of audio for speaker embedding

## License

MIT
