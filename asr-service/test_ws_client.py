#!/usr/bin/env python3
"""Quick test of WebSocket connection to ASR server."""

import asyncio
import json
import websockets
import numpy as np


async def test_connection():
    url = "ws://127.0.0.1:8765"
    print(f"Connecting to {url}...")

    async with websockets.connect(url) as ws:
        print("Connected!")

        # Test ping
        await ws.send(json.dumps({"type": "ping"}))
        print("Sent: ping")
        response = await ws.recv()
        print(f"Received: {response}")

        # Start session
        await ws.send(json.dumps({"type": "start"}))
        print("Sent: start")
        response = await ws.recv()
        print(f"Received: {response}")

        # Send some silent audio (should not produce transcription)
        print("\nSending 3 seconds of silent audio...")
        sample_rate = 16000
        duration = 3.0
        samples = int(sample_rate * duration)
        # Generate silent audio (zeros)
        audio = np.zeros(samples, dtype=np.int16)
        await ws.send(audio.tobytes())
        print(f"Sent {len(audio.tobytes())} bytes of audio")

        # Check for any response (transcription or error)
        try:
            response = await asyncio.wait_for(ws.recv(), timeout=5.0)
            print(f"Received: {response}")
        except asyncio.TimeoutError:
            print("No transcription response (expected for silence)")

        # Stop session
        await ws.send(json.dumps({"type": "stop"}))
        print("\nSent: stop")
        response = await ws.recv()
        print(f"Received: {response}")

        print("\nConnection test successful!")


if __name__ == "__main__":
    asyncio.run(test_connection())
