import json
import sys
import subprocess
import typing as tp
from io import BytesIO
import logging

import ffmpeg
import lameenc
import numpy as np
import torch
import torchaudio as ta
import soundfile as sf

from demucs.audio import convert_audio, convert_audio_channels, prevent_clip, i16_pcm

logger = logging.getLogger('uvicorn.error')

class AudioFileContent:
    def __init__(self, file_bytes: bytes, extension: str):
        self._file_bytes = file_bytes
        self.extension = extension.lower()
        self._info = self._read_info_from_bytes()

    def _read_info_from_bytes(self):
        command = [
            "ffprobe",
            "-v", "error",
            "-f", self.extension,      # format
            "-i", "pipe:0",       # read from stdin
            "-show_format",
            "-show_streams",
            "-print_format", "json"
        ]
        process = subprocess.run(
            command,
            input=self._file_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        if process.returncode != 0:
            raise RuntimeError(f"ffprobe error: {process.stderr.decode()}")

        return json.loads(process.stdout.decode())

    @property
    def info(self):
        return self._info

    @property
    def duration(self):
        return float(self.info['format']['duration'])

    @property
    def _audio_streams(self):
        return [
            index for index, stream in enumerate(self.info["streams"])
            if stream["codec_type"] == "audio"
        ]

    def __len__(self):
        return len(self._audio_streams)

    def channels(self, stream=0):
        return int(self.info['streams'][self._audio_streams[stream]]['channels'])

    def samplerate(self, stream=0):
        return int(self.info['streams'][self._audio_streams[stream]]['sample_rate'])

    def read(self,
             seek_time=None,
             duration=None,
             streams=slice(None),
             samplerate=None,
             channels=None):
        streams = np.array(range(len(self)))[streams]
        single = not isinstance(streams, np.ndarray)
        if single:
            streams = [streams]

        target_size = None
        if duration is not None:
            sr = samplerate or self.samplerate()
            target_size = int(sr * duration)

        wavs = []
        for stream_idx in streams:
            input_kwargs = {
                'format': self.extension,
                'loglevel': 'panic'
            }
            output_kwargs = {
                'format': 'f32le',
                'ac': str(self.channels(stream_idx)),
                'ar': str(samplerate or self.samplerate())
            }
            if seek_time:
                output_kwargs['ss'] = str(seek_time)
            if duration:
                output_kwargs['t'] = str(duration)

            process = (
                ffmpeg
                .input('pipe:0', **input_kwargs)
                .output('pipe:1', **output_kwargs)
                .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True)
            )
            out, err = process.communicate(input=self._file_bytes)
            wav = np.frombuffer(out, dtype=np.float32).copy()
            wav = torch.from_numpy(wav).view(-1, self.channels(stream_idx)).t()
            if channels is not None:
                wav = convert_audio_channels(wav, channels)
            if target_size is not None:
                wav = wav[..., :target_size]
            wavs.append(wav)

        wav = torch.stack(wavs, dim=0)
        return wav[0] if single else wav
    
def load_wav(audio_bytes, extension, audio_channels, samplerate, seek_time, duration):
    errors = {}
    wav = None

    try:
        wav = AudioFileContent(file_bytes=audio_bytes, extension=extension).read(
            seek_time=seek_time,
            duration=duration,
            streams=0,
            samplerate=samplerate,
            channels=audio_channels)
    except FileNotFoundError:
        errors['ffmpeg'] = 'FFmpeg is not installed.'
    except subprocess.CalledProcessError:
        errors['ffmpeg'] = 'FFmpeg could not read the file.'

    if wav is None:
        logger.warning("Attempting to load audio with torchaudio because wav is None.")
        try:
            wav, sr = ta.load(BytesIO(audio_bytes))
        except RuntimeError as err:
            errors['torchaudio'] = err.args[0]
        else:
            wav = convert_audio(wav, sr, samplerate, audio_channels)

    if wav is None:
        logger.error(f"Could not load audio bytes {audio_bytes}.")
        for backend, error in errors.items():
            logger.error(f"When trying to load using {backend}, got the following error: {error}")
        sys.exit(1)
    return wav

def convert_audio_wav_to_bytes(
    wav: torch.Tensor,
    format: tp.Literal["wav", "flac", "mp3", "ogg"],
    samplerate: int,
    clip: tp.Literal["rescale", "clamp", "tanh", "none"] = 'rescale',
) -> bytes:
    wav = prevent_clip(wav, mode=clip)
    buf = BytesIO()
    wav_np = wav.cpu().numpy().T

    fmt = format.upper()
    if fmt == "WAV":
        subtype = 'PCM_16'
    elif fmt == "FLAC":
        subtype = 'PCM_16'
    elif fmt == "MP3":
        subtype = 'MPEG_LAYER_III'
    elif fmt == "OGG":
        subtype = 'VORBIS'
    else:
        subtype = None

    sf.write(buf, wav_np, samplerate, format=fmt, subtype=subtype)
    return buf.getvalue()
