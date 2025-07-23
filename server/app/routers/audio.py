import asyncio
import io
import logging
import uuid
import time
import pickle
import json
from enum import Enum
from urllib.parse import quote
from typing import List, Literal

from fastapi import APIRouter, UploadFile, HTTPException, File, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.internal.services.audio import AudioSeparator, SeparationStreamProgress, AudioTranscriber, TranscriptionStreamProgress
from app.config import settings

class SeparationEventType(Enum):
    DONE = "done"
    STEM = "stem"
    PROGRESS = "progress"
    STATUS = "status"
    MESSAGE = "message"
    UNKNOWN = "unknown"

class TranscriptionEventType(str, Enum):
    STATUS = "status"
    LANGUAGE = "language"
    SEGMENT = "segment"
    DONE = "done"

class StemMetadata(BaseModel):
    id: str
    name: str
    size: int

class SeparateAudioSourcesResponse(BaseModel):
    stems: List[StemMetadata]

class TranscriptionWordSegment(BaseModel):
    start: float
    end: float
    word: str

class TranscriptionTextSegment(BaseModel):
    start: float
    end: float
    text: str
    words: List[TranscriptionWordSegment]

class TranscribeAudioResponse(BaseModel):
    segments: List[TranscriptionTextSegment]
    language: str

logger = logging.getLogger("uvicorn.error")

router = APIRouter(
    prefix="/audio",
    tags=["audio"],
)

separator = AudioSeparator(settings.separation_model)

in_memory_stems: dict = {}  # id -> (filename, BytesIO, mime_type, timestamp)

async def stem_cleanup_coroutine(stem_ttl_seconds: int, cleanup_interval: int):
    while True:
        now = time.time()
        to_delete = [sid for sid, (_, _, _, ts) in in_memory_stems.items() if now - ts > stem_ttl_seconds]
        for sid in to_delete:
            filename, *_ = in_memory_stems.pop(sid)
            logger.info(f"[CLEANUP] Stem expired: {filename} (ID: {sid})")
        await asyncio.sleep(cleanup_interval)

@router.post("/separate", response_model=SeparateAudioSourcesResponse)
async def separate_audio_sources(
    file: UploadFile = File(...),
    stem: Literal["drum", "bass", "other", "vocals"] = Form(None),
    output_format: Literal["wav", "flac", "mp3", "ogg"] = Form("flac"),
    start: str = Form(None),
    end: str = Form(None),
):
    file_bytes = await file.read()
    extension = file.filename.split('.')[-1].lower()

    stem = None if stem not in ["drum", "bass", "other", "vocals"] else stem
    seek_time = None
    duration = None
    start = None if start.strip() == "" else start
    end = None if end.strip() == "" else end
    if start is not None and end is not None:
        seek_time = float(start)
        duration = float(end) - float(start)

    stems = separator.separate(
        audio_bytes=file_bytes,
        input_format=extension,
        stem=stem,
        output_format=output_format,
        seek_time=seek_time,
        duration=duration
    )

    response_data = []
    now = time.time()

    for name, audio_bytes in stems.items():
        uid = str(uuid.uuid4())
        memory_file = io.BytesIO(audio_bytes)
        mime_type = f"audio/{output_format}"
        in_memory_stems[uid] = (f"{file.filename.split('.')[-2]}_{name}.{output_format}", memory_file, mime_type, now)

        response_data.append({
            "id": uid,
            "name": f"{name}.{output_format}",
            "size": len(audio_bytes)
        })
        response_data.append(StemMetadata(id=uid, name=f"{name}.{output_format}", size=len(audio_bytes)))

    return SeparateAudioSourcesResponse(stems=response_data)

# TODO: Handle client disconnecting
@router.post("/separate/stream")
async def separate_audio_sources_stream(
    file: UploadFile = File(...),
    stem: Literal["drum", "bass", "other", "vocals"] = Form(None),
    output_format: Literal["wav", "flac", "mp3", "ogg"] = Form("flac"),
    start: str = Form(None),
    end: str = Form(None),
):
    """
    Stream audio source separation events using Server-Sent Events (SSE).

    This endpoint separates an uploaded audio file into individual stems
    (e.g., vocals, drums, bass) and streams real-time updates to the client using SSE.

    Supported SSE event types and their payloads:

    - **status**:
        Plain text describing current phase of processing, including: LOADING_WAVEFORM, APPLYING_MODEL, CONVERTING_SOURCES.
        ```
        data: "LOADING_WAVEFORM"
        ```

    - **progress**:
        JSON object indicating current processing progress.
        ```
        {
            "processed": 2,        // number of segments processed
            "total": 10,           // total number of segments
            "eta": "00:01:23"      // estimated time remaining in HH:MM:SS format, omit empty
        }
        ```

    - **stem**:
        JSON object with metadata of a completed stem.
        ```
        {
            "id": "uuid-string",
            "name": "vocals.wav",
            "size": 123456         // size in bytes
        }
        ```

    - **done**:
        Signals that all stems have been generated and streamed.
        ```
        data: null
        ```

    Parameters:
    - **file**: The uploaded audio file (formats like MP3, WAV, FLAC are supported).
    - **stem** (optional): Specific stem to extract (`"vocals"`, `"drum"`, `"bass"`, `"other"`). Defaults to all.
    - **output_format**: Desired output format. Default is `"flac"`.
    - **start** (optional): Start time in seconds for partial separation.
    - **end** (optional): End time in seconds for partial separation.

    Returns:
        StreamingResponse: An event stream using `text/event-stream` format with real-time updates.
    """

    file_bytes = await file.read()
    extension = file.filename.split('.')[-1].lower()
    stem = None if stem not in ["drum", "bass", "other", "vocals"] else stem
    seek_time = None
    duration = None
    start = None if start is None or start.strip() == "" else start
    end = None if end is None or end.strip() == "" else end
    if start and end:
        seek_time = float(start)
        duration = float(end) - float(start)

    def event_stream():
        for chunk in separator.separate_streaming(
            audio_bytes=file_bytes,
            input_format=extension,
            stem=stem,
            output_format=output_format,
            seek_time=seek_time,
            duration=duration
        ):
            if isinstance(chunk, bytes):
                # Final result payload
                if chunk.startswith(SeparationStreamProgress.END.value):
                    stems = pickle.loads(chunk[len(SeparationStreamProgress.END.value):])
                    for name, audio_bytes in stems.items():
                        uid = str(uuid.uuid4())
                        now = time.time()
                        memory_file = io.BytesIO(audio_bytes)
                        mime_type = f"audio/{output_format}"
                        in_memory_stems[uid] = (
                            f"{file.filename.split('.')[-2]}_{name}.{output_format}",
                            memory_file,
                            mime_type,
                            now,
                        )
                        meta = {
                            "id": uid,
                            "name": f"{name}.{output_format}",
                            "size": len(audio_bytes)
                        }
                        yield f"event: {SeparationEventType.STEM.value}\ndata: {json.dumps(meta)}\n\n"

                    yield f"event: {SeparationEventType.DONE.value}\ndata: null\n\n"
                # Segment progress update
                elif chunk.startswith(SeparationStreamProgress.PROGRESS.value):
                    payload = pickle.loads(chunk[len(SeparationStreamProgress.PROGRESS.value):])
                    yield f"event: {SeparationEventType.PROGRESS.value}\ndata: {json.dumps(payload)}\n\n"
                else:
                    # Unknown binary chunk
                    yield f"event: {SeparationEventType.MESSAGE.value}\ndata: {chunk.decode(errors='ignore')}\n\n"
            else:
                # Plain string progress (e.g., LOADING_WAVEFORM, APPLYING_MODEL, etc.)
                yield f"event: {SeparationEventType.STATUS.value}\ndata: {chunk.strip()}\n\n"

    headers = {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
    }

    return StreamingResponse(event_stream(), headers=headers)

@router.get("/get/{stem_id}")
async def get_stem(stem_id: str):
    """
    Retrieve and download a previously separated audio stem by its unique ID.

    The stem is served as an audio file in the format requested during separation (e.g., WAV or MP3).
    After the file is served, it is removed from in-memory storage.

    Response headers:
    - `Content-Disposition: inline; filename="{stem_name}"` – suggests a download filename.
    - `Content-Type: audio/{format}` – MIME type based on the requested format (e.g., audio/wav, audio/mp3).

    Parameters:
    - **stem_id**: A UUID string identifying the requested stem.

    Returns:
        StreamingResponse: The binary audio file.

    Raises:
        404 Not Found: If the `stem_id` does not exist or has expired.
    """

    if stem_id not in in_memory_stems:
        raise HTTPException(status_code=404, detail="Stem not found")

    filename, memory_file, mime_type, _ = in_memory_stems.pop(stem_id)
    memory_file.seek(0)
    logger.info(f"[CLEANUP] Stem served: {filename} (ID: {stem_id})")

    return StreamingResponse(memory_file, media_type=mime_type, headers={
        "Content-Disposition": f"inline; filename*=UTF-8''{quote(filename)}"
    })

@router.post("/transcribe", response_model=TranscribeAudioResponse)
async def transcribe_audio(
    file: UploadFile = File(...),
    language: str = Form(None),
):
    file_bytes = await file.read()
    extension = file.filename.split('.')[-1].lower()

    # TODO: Transcriber occupies all GPU memory, so load it only when needed. Should declare it as a global variable when have enough resources.
    transcriptor = AudioTranscriber(settings.transcription_model)

    transcription = transcriptor.transcribe(
        model=transcriptor,
        audio_bytes=file_bytes,
        format=extension,
        task="transcribe",
        language=language,
        word_timestamps=True,
        verbose=True
    )

    return TranscribeAudioResponse(
        segments=transcription["segments"],
        language=transcription["language"]
    )

# TODO: Handle client disconnecting
@router.post("/transcribe/stream")
async def transcribe_audio_stream(
    file: UploadFile = File(...),
    language: str = Form(None),
):
    """
    Stream transcription results using Server-Sent Events (SSE).

    Supported SSE event types:

    - **status**:
        Plain string for current stage (e.g. "LOADING_WAVEFORM", "LANGUAGE", "TRANSCRIBING_WAVEFORM")
        ```
        data: "TRANSCRIBING_WAVEFORM"
        ```

    - **segment**:
        JSON object with real-time transcribed segment info.
        ```
        {
            "start": 1.2,
            "end": 4.3,
            "text": "hello world",
            "words": [ ... ]  // optional word timestamps
        }
        ```

    - **done**:
        Final event with full transcript, all segments, and language.
        ```
        {
            "text": "complete transcript...",
            "segments": [...],
            "language": "en"
        }
        ```

    Returns:
        StreamingResponse: SSE-compatible event stream.
    """

    file_bytes = await file.read()
    extension = file.filename.split('.')[-1].lower()

    # TODO: Transcriber occupies all GPU memory, so load it only when needed. Should declare it as a global variable when have enough resources.
    transcriptor = AudioTranscriber(settings.transcription_model)

    def event_stream():
        for chunk in transcriptor.transcribe_streaming(
            model=transcriptor,
            audio_bytes=file_bytes,
            format=extension,
            task="transcribe",
            language=language,
            word_timestamps=True,
        ):
            if isinstance(chunk, str):
                yield f"event: {TranscriptionEventType.STATUS.value}\ndata: {chunk.strip()}\n\n"
            elif isinstance(chunk, bytes):
                if chunk.startswith(TranscriptionStreamProgress.SEGMENT.value):
                    payload = pickle.loads(chunk[len(TranscriptionStreamProgress.SEGMENT.value):])
                    yield f"event: {TranscriptionEventType.SEGMENT.value}\ndata: {json.dumps(payload)}\n\n"
                elif chunk.startswith(TranscriptionStreamProgress.LANGUAGE.value):
                    payload = pickle.loads(chunk[len(TranscriptionStreamProgress.LANGUAGE.value):])
                    yield f"event: {TranscriptionEventType.LANGUAGE.value}\ndata: {json.dumps(payload)}\n\n"
                elif chunk.startswith(TranscriptionStreamProgress.END.value):
                    payload = pickle.loads(chunk[len(TranscriptionStreamProgress.END.value):])
                    yield f"event: {TranscriptionEventType.DONE.value}\ndata: {json.dumps(payload)}\n\n"
            else:
                # fallback case (optional)
                yield f"event: {TranscriptionEventType.STATUS.value}\ndata: {str(chunk).strip()}\n\n"

    headers = {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
    }

    return StreamingResponse(event_stream(), headers=headers)
