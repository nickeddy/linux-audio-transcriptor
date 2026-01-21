"""Main entry point for ASR service."""

import argparse
import asyncio
import logging
import sys

from .server import ASRServer
from .streaming_transcriber import StreamingTranscriber
from .transcriber import NemotronTranscriber  # For file testing


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Real-time ASR service using NVIDIA Nemotron",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind the server to",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port to bind the server to",
    )
    parser.add_argument(
        "--model",
        default="nvidia/nemotron-speech-streaming-en-0.6b",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=160,
        help="Chunk size in milliseconds for streaming (80, 160, 560, or 1120)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--test-file",
        metavar="FILE",
        help="Transcribe a single audio file and exit (for testing)",
    )

    return parser.parse_args()


async def run_server(args: argparse.Namespace) -> None:
    """Run the ASR WebSocket server."""
    transcriber = StreamingTranscriber(
        model_name=args.model,
        chunk_size_ms=args.chunk_size,
    )

    server = ASRServer(
        host=args.host,
        port=args.port,
        transcriber=transcriber,
    )

    print(f"Starting ASR server on ws://{args.host}:{args.port}")
    print(f"Model: {args.model}")
    print(f"Chunk size: {args.chunk_size}ms (streaming mode)")
    print("Press Ctrl+C to stop")
    print()

    try:
        await server.run_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        await server.stop()


def test_transcribe_file(args: argparse.Namespace) -> None:
    """Test transcription with a single file."""
    transcriber = NemotronTranscriber(
        model_name=args.model,
    )

    print(f"Loading model: {args.model}")
    transcriber.load_model()

    print(f"Transcribing: {args.test_file}")
    result = transcriber.transcribe_file(args.test_file)

    print("\n--- Transcription ---")
    print(result)
    print("--- End ---")


def main() -> None:
    """Main entry point."""
    args = parse_args()
    setup_logging(args.verbose)

    # Test mode: transcribe a single file
    if args.test_file:
        test_transcribe_file(args)
        return

    # Server mode
    try:
        asyncio.run(run_server(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
