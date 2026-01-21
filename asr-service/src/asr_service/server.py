"""WebSocket server for streaming ASR."""

import asyncio
import json
import logging
import struct
from dataclasses import asdict
from typing import Any

import numpy as np
import websockets
from websockets.server import WebSocketServerProtocol

from .streaming_transcriber import StreamingTranscriber, StreamingResult

logger = logging.getLogger(__name__)


class ASRServer:
    """WebSocket server for real-time streaming ASR.

    Protocol:
        - Client sends binary audio data (16-bit PCM, 16kHz mono)
        - Server responds with JSON transcription results
        - Control messages are JSON with "type" field

    Transcription messages:
        {"type": "partial", "text": "...", "segment_id": 0}  - Interim result
        {"type": "final", "text": "...", "segment_id": 0}    - Final result for segment

    Control messages:
        {"type": "start"} - Start a new transcription session
        {"type": "stop"} - End the current session
        {"type": "finalize"} - Finalize current utterance (on silence)
        {"type": "ping"} - Health check, responds with {"type": "pong"}
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8765,
        transcriber: StreamingTranscriber | None = None,
    ):
        """Initialize the ASR server.

        Args:
            host: Bind address.
            port: Bind port.
            transcriber: Transcriber instance. Created if not provided.
        """
        self.host = host
        self.port = port
        self.transcriber = transcriber or StreamingTranscriber()
        self._server = None
        self._active_sessions: dict[str, bool] = {}

    async def start(self) -> None:
        """Start the WebSocket server."""
        # Load model before accepting connections
        logger.info("Loading ASR model...")
        self.transcriber.load_model()
        logger.info("Model loaded, starting server...")

        self._server = await websockets.serve(
            self._handle_client,
            self.host,
            self.port,
        )
        logger.info(f"ASR server listening on ws://{self.host}:{self.port}")

    async def stop(self) -> None:
        """Stop the server."""
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            logger.info("ASR server stopped")

    async def run_forever(self) -> None:
        """Start server and run until interrupted."""
        await self.start()
        await self._server.wait_closed()

    async def _handle_client(
        self,
        websocket: WebSocketServerProtocol,
    ) -> None:
        """Handle a single client connection."""
        client_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        logger.info(f"Client connected: {client_id}")
        self._active_sessions[client_id] = False

        try:
            async for message in websocket:
                response = await self._process_message(client_id, message)
                if response:
                    await websocket.send(json.dumps(response))
        except websockets.ConnectionClosed:
            logger.info(f"Client disconnected: {client_id}")
        except Exception as e:
            logger.error(f"Error handling client {client_id}: {e}")
            await websocket.send(json.dumps({
                "type": "error",
                "message": str(e),
            }))
        finally:
            # Clean up session
            if self._active_sessions.get(client_id):
                self.transcriber.end_session()
                self._active_sessions[client_id] = False
            self._active_sessions.pop(client_id, None)

    async def _process_message(
        self,
        client_id: str,
        message: bytes | str,
    ) -> dict[str, Any] | None:
        """Process a message from a client.

        Args:
            client_id: Client identifier.
            message: Raw message (binary audio or JSON control).

        Returns:
            Response dict or None.
        """
        # Handle control messages (JSON strings)
        if isinstance(message, str):
            return await self._handle_control_message(client_id, message)

        # Handle binary audio data
        if isinstance(message, bytes):
            return await self._handle_audio_data(client_id, message)

        return {"type": "error", "message": "Unknown message format"}

    async def _handle_control_message(
        self,
        client_id: str,
        message: str,
    ) -> dict[str, Any]:
        """Handle a JSON control message."""
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            return {"type": "error", "message": "Invalid JSON"}

        msg_type = data.get("type")

        if msg_type == "ping":
            return {"type": "pong"}

        elif msg_type == "start":
            self.transcriber.start_session()
            self._active_sessions[client_id] = True
            logger.info(f"Started session for {client_id}")
            return {"type": "started"}

        elif msg_type == "stop":
            response_data = {"type": "stopped"}
            if self._active_sessions.get(client_id):
                # End session and get any final text
                result = self.transcriber.end_session()
                if result and result.text:
                    response_data["final_text"] = result.text
                    response_data["segment_id"] = result.segment_id
                self._active_sessions[client_id] = False
            logger.info(f"Stopped session for {client_id}")
            return response_data

        elif msg_type == "finalize":
            # Finalize current utterance (called when VAD detects silence)
            logger.info(f"Received finalize request from {client_id}")
            if self._active_sessions.get(client_id):
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    self.transcriber.finalize_segment,
                )
                logger.info(f"Finalize result: {result}")
                if result and result.text:
                    return {
                        "type": "final",
                        "text": result.text,
                        "segment_id": result.segment_id,
                        "speaker": result.speaker,
                    }
            return None

        elif msg_type == "status":
            return {
                "type": "status",
                "model_loaded": self.transcriber.is_loaded,
                "session_active": self._active_sessions.get(client_id, False),
            }

        else:
            return {"type": "error", "message": f"Unknown message type: {msg_type}"}

    async def _handle_audio_data(
        self,
        client_id: str,
        data: bytes,
    ) -> dict[str, Any] | None:
        """Handle binary audio data."""
        if not self._active_sessions.get(client_id):
            return {
                "type": "error",
                "message": "No active session. Send {\"type\": \"start\"} first.",
            }

        # Convert bytes to numpy array (16-bit PCM)
        audio_chunk = np.frombuffer(data, dtype=np.int16)

        # Stream audio through the transcriber
        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self.transcriber.add_audio,
            audio_chunk,
        )

        if result and result.text:
            # Return partial or final based on result
            return {
                "type": "partial" if result.is_partial else "final",
                "text": result.text,
                "segment_id": result.segment_id,
                "speaker": result.speaker,
            }

        # No new transcription available
        return None


async def run_server(
    host: str = "127.0.0.1",
    port: int = 8765,
) -> None:
    """Run the ASR server."""
    server = ASRServer(host=host, port=port)
    await server.run_forever()
