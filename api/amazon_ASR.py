import asyncio
import aiofile
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent
from amazon_transcribe.utils import apply_realtime_delay

SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2
CHANNEL_NUMS = 1
CHUNK_SIZE = 1024 * 8
REGION = "us-east-2"

class MyEventHandler(TranscriptResultStreamHandler):
    def __init__(self, output_stream):
        super().__init__(output_stream)
        self.transcripts = []

    async def handle_transcript_event(self, transcript_event: TranscriptEvent):
        results = transcript_event.transcript.results
        for result in results:
            for alt in result.alternatives:
                self.transcripts.append(alt.transcript)

def read_audio_chunks(file_path, chunk_size):
    with open(file_path, "rb") as audio_file:
        while True:
            data = audio_file.read(chunk_size)
            if not data:
                break
            yield data

def amazon_ASR(audio_file_path):
    async def transcribe():
        # Setup up our client with our chosen AWS region
        client = TranscribeStreamingClient(region=REGION)

        # Start transcription to generate our async stream
        stream = await client.start_stream_transcription(
            language_code="en-US",
            media_sample_rate_hz=SAMPLE_RATE,
            media_encoding="pcm",
        )

        async def write_chunks():
            async with aiofile.AIOFile(audio_file_path, "rb") as afp:
                reader = aiofile.Reader(afp, chunk_size=CHUNK_SIZE)
                await apply_realtime_delay(
                    stream, reader, BYTES_PER_SAMPLE, SAMPLE_RATE, CHANNEL_NUMS
                )
            await stream.input_stream.end_stream()

        # Instantiate our handler and start processing events
        handler = MyEventHandler(stream.output_stream)
        await asyncio.gather(write_chunks(), handler.handle_events())
        return handler.transcripts

    transcripts = asyncio.run(transcribe())


    return transcripts[-1]

if __name__ == "__main__":
    audio_file_path = "./benign_audio/p225_001.wav"

    transcription_result = amazon_asr(audio_file_path)
    print(transcription_result)
