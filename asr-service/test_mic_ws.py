#!/usr/bin/env python3
"""Test microphone capture with WebSocket ASR server.

Requires: pip install sounddevice
"""

import asyncio
import json
import queue
import sys
import numpy as np
import websockets

try:
    import sounddevice as sd
except ImportError:
    print("sounddevice not installed. Install with: pip install sounddevice")
    sys.exit(1)


SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_DURATION_MS = 100  # Send audio every 100ms

audio_queue = queue.Queue()


def audio_callback(indata, frames, time, status):
    """Called by sounddevice for each audio chunk."""
    if status:
        print(f"Audio status: {status}", file=sys.stderr)
    # Convert to int16 and queue
    audio_int16 = (indata[:, 0] * 32767).astype(np.int16)
    audio_queue.put(audio_int16.tobytes())


async def main():
    print("=== Microphone to ASR Test (Python) ===\n")

    url = "ws://127.0.0.1:8765"
    print(f"Connecting to {url}...")

    async with websockets.connect(url) as ws:
        print("Connected!")

        # Start session
        await ws.send(json.dumps({"type": "start"}))
        response = await ws.recv()
        print(f"Server: {response}")

        print("\nStarting microphone capture at 16kHz...")
        print("Speak now! Press Ctrl+C to stop.\n")
        print("-------------------------------------------")

        # Start audio capture
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype='float32',
            callback=audio_callback,
            blocksize=int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000),
        )

        # Task to receive transcriptions
        async def receive_transcriptions():
            try:
                while True:
                    msg = await asyncio.wait_for(ws.recv(), timeout=0.1)
                    data = json.loads(msg)
                    if data.get("type") == "transcription":
                        text = data.get("text", "")
                        if text:
                            print(f">> {text}")
            except asyncio.TimeoutError:
                pass
            except websockets.ConnectionClosed:
                return
            except Exception as e:
                print(f"Receive error: {e}")

        try:
            with stream:
                while True:
                    # Send any queued audio
                    try:
                        while not audio_queue.empty():
                            audio_bytes = audio_queue.get_nowait()
                            await ws.send(audio_bytes)
                    except queue.Empty:
                        pass

                    # Check for transcriptions
                    await receive_transcriptions()
                    await asyncio.sleep(0.05)

        except KeyboardInterrupt:
            print("\n-------------------------------------------")
            print("\nStopping...")

        # Stop session and get final transcription
        await ws.send(json.dumps({"type": "stop"}))
        response = await ws.recv()
        data = json.loads(response)
        print(f"Server: {response}")
        if data.get("final_text"):
            print(f"Final transcription: {data['final_text']}")

        print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
