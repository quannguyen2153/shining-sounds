from typing import Literal, Optional, Union
from subprocess import run, CalledProcessError

import numpy as np
import torch
import torch.nn.functional as F
from whisper.audio import mel_filters, SAMPLE_RATE, N_FFT, HOP_LENGTH

def load_audio_bytes(audio_bytes: bytes, format: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Decode an audio file from bytes and return a mono waveform, resampled if needed.

    Parameters
    ----------
    audio_bytes : bytes
        Raw audio file content in bytes.
    format : str
        Format of the input audio (e.g. "mp3", "wav", "flac").
    sr : int, optional
        Target sample rate to resample the audio. Default is 44100.

    Returns
    -------
    np.ndarray
        A NumPy array containing the mono audio waveform in float32 format.
    """
    cmd = [
        "ffmpeg",
        "-f", format,
        "-i", "pipe:0",          # read from stdin
        "-f", "s16le",           # output raw 16-bit PCM
        "-ac", "1",              # mono
        "-acodec", "pcm_s16le",  # raw audio codec
        "-ar", str(sr),          # sample rate
        "-"
    ]

    try:
        result = run(cmd, input=audio_bytes, capture_output=True, check=True)
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to decode audio: {e.stderr.decode()}") from e

    return np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0

def log_mel_spectrogram(
    audio_bytes: bytes,
    format: Literal["mp3", "wav", "flac"],
    n_mels: int = 80,
    padding: int = 0,
    device: Optional[Union[str, torch.device]] = None,
):
    """
    Compute the log-Mel spectrogram of

    Parameters
    ----------
    audio_bytes: bytes
        The raw audio file content in bytes.

    format: str
        The format of the input audio (e.g. "mp3", "wav", "flac").

    n_mels: int
        The number of Mel-frequency filters, only 80 and 128 are supported

    padding: int
        Number of zero samples to pad to the right

    device: Optional[Union[str, torch.device]]
        If given, the audio tensor is moved to this device before STFT

    Returns
    -------
    torch.Tensor, shape = (n_mels, n_frames)
        A Tensor that contains the Mel spectrogram
    """
    
    audio = torch.from_numpy(load_audio_bytes(audio_bytes, format))

    if device is not None:
        audio = audio.to(device)
    if padding > 0:
        audio = F.pad(audio, (0, padding))
    window = torch.hann_window(N_FFT).to(audio.device)
    stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2

    filters = mel_filters(audio.device, n_mels)
    mel_spec = filters @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec