import time
import typing as tp
import logging
import pickle
import random
from enum import Enum
from concurrent.futures import ThreadPoolExecutor

from dora.log import fatal
import torch

from demucs.apply import apply_model, BagOfModels, TensorChunk, Model, tensor_chunk
from demucs.htdemucs import HTDemucs
from demucs.pretrained import get_model, ModelLoadingError, DEFAULT_MODEL
from demucs.utils import DummyPoolExecutor, center_trim

from .utils import load_wav, convert_audio_wav_to_bytes

logger = logging.getLogger('uvicorn.error')

class Separator:
    class SeparationStreamProgress(Enum):
        LOADING_WAVEFORM = b"LOADING_WAVEFORM"
        APPLYING_MODEL = b"APPLYING_MODEL"
        PROGRESS = b"PROGRESS"
        CONVERTING_SOURCES = b"CONVERTING_SOURCES"
        END = b"END"

    def __init__(self, model_name=DEFAULT_MODEL, model_repo=None, device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the Separator with a specific model.

        Args:
            model_name (str, optional): Name of the separation model to use.
            model_repo (str, optional): Optional model repository path or URL.
            device (str, optional): Device to run the model on ("cuda" or "cpu").
        """
        self.device = device
        logger.info(f"Using device: {self.device}")

        try:
            self.model = get_model(name=model_name, repo=model_repo)
        except ModelLoadingError as error:
            fatal(error.args[0])

        if isinstance(self.model, BagOfModels):
            logger.info(f"Selected model is a bag of {len(self.model.models)} models. You will see that many progress bars per track.")

        self.model.to(device)
        self.model.eval()

    def separate(
        self,
        audio_bytes: bytes,
        input_format: tp.Literal["mp3", "wav", "flac"],
        output_format: tp.Literal["mp3", "wav"],
        seek_time: tp.Optional[float] = None,
        duration: tp.Optional[float] = None,
        shifts=1,
        overlap=0.25,
        split=True,
        segment=None,
        jobs=0,
        stem=None,
        clip_mode: tp.Literal["rescale", "clamp", "tanh", "none"] = "rescale",
        wav_output_type: tp.Literal["int16", "int24", "float32"] = "int16",
        mp3_bitrate=320,
        mp3_preset=2,
    ) -> dict[str, bytes]:
        """
        Perform source separation on raw audio content in bytes.

        Args:
            audio_bytes (bytes): Raw audio content in bytes.
            input_format (Literal["mp3", "wav", "flac"]): Format of the input audio.
            output_format (Literal["mp3", "wav"], optional): Format of the output audio.
            seek_time (float, optional): Start time in seconds to begin processing the audio.
            duration (float, optional): Duration in seconds to process from the seek time.
            shifts (int, optional): Number of shifted versions to use for test-time ensembling.
            overlap (float, optional): Overlap between audio segments for smoother separation.
            split (bool, optional): Whether to split long audio into segments.
            segment (float, optional): Segment length (in seconds) if using split; must not exceed model limit.
            jobs (int, optional): Number of parallel jobs for CPU processing.
            stem (str, optional): Specific stem to isolate (e.g., "vocals"); returns that and the complementary mix.
            clip_mode (str, optional): Strategy for post-processing audio amplitude ("rescale", "clamp", "tanh", "none").
            wav_output_type (Literal["int16", "int24", "float32"], optional): Format of the output waveform data.
            mp3_bitrate (int, optional): Bitrate in kbps for MP3 encoding.
            mp3_preset (int, optional): Preset level for MP3 encoding (lower is slower but better quality).

        Returns:
            dict[str, bytes]: A dictionary mapping stem names (e.g., "vocals", "drums", "no_vocals") to byte-encoded audio files.
        """

        max_allowed_segment = float('inf')
        if isinstance(self.model, HTDemucs):
            max_allowed_segment = float(self.model.segment)
        elif isinstance(self.model, BagOfModels):
            max_allowed_segment = self.model.max_allowed_segment
        if segment is not None and segment > max_allowed_segment:
            fatal("Cannot use a Transformer model with a longer segment "
                f"than it was trained for. Maximum segment is: {max_allowed_segment}")

        if stem is not None and stem not in self.model.sources:
            fatal(
                'error: stem "{stem}" is not in selected model. STEM must be one of {sources}.'.format(
                    stem=stem, sources=', '.join(self.model.sources)))
        
        
        wav = load_wav(audio_bytes, input_format, self.model.audio_channels, self.model.samplerate, seek_time, duration)

        ref = wav.mean(0)
        wav -= ref.mean()
        wav /= ref.std()
        sources = apply_model(self.model, wav[None], device=self.device, shifts=shifts,
                                split=split, overlap=overlap, progress=True,
                                num_workers=jobs, segment=segment)[0]
        sources *= ref.std()
        sources += ref.mean()

        kwargs = {
            'samplerate': self.model.samplerate,
            'bitrate': mp3_bitrate,
            'preset': mp3_preset,
            'clip': clip_mode,
            'as_float': wav_output_type == "float32",
            'bits_per_sample': 24 if wav_output_type == "int24" else 16,
        }
        result = {}

        if stem is None:
            for source, name in zip(sources, self.model.sources):
                result[name] = convert_audio_wav_to_bytes(source, output_format, **kwargs)
        else:
            sources = list(sources)
            result[stem] = convert_audio_wav_to_bytes(sources.pop(self.model.sources.index(stem)), output_format, **kwargs)
            # Warning : after poping the stem, selected stem is no longer in the list 'sources'
            other_stem = torch.zeros_like(sources[0])
            for i in sources:
                other_stem += i
            result["no_"+stem] = convert_audio_wav_to_bytes(other_stem, output_format, **kwargs)

        return result
    
    def separate_streaming(
        self,
        audio_bytes: bytes,
        input_format: tp.Literal["mp3", "wav", "flac"],
        output_format: tp.Literal["mp3", "wav"],
        seek_time: tp.Optional[float] = None,
        duration: tp.Optional[float] = None,
        shifts=1,
        overlap=0.25,
        split=True,
        segment=None,
        jobs=0,
        stem=None,
        clip_mode: tp.Literal["rescale", "clamp", "tanh", "none"] = "rescale",
        wav_output_type: tp.Literal["int16", "int24", "float32"] = "int16",
        mp3_bitrate=320,
        mp3_preset=2,
    ):
        yield self.SeparationStreamProgress.LOADING_WAVEFORM.name
        wav = load_wav(audio_bytes, input_format, self.model.audio_channels, self.model.samplerate, seek_time, duration)

        ref = wav.mean(0)
        wav -= ref.mean()
        wav /= ref.std()

        yield self.SeparationStreamProgress.APPLYING_MODEL.name
        apply_gen = self.apply_model_streaming(
            self.model, wav[None], device=self.device, shifts=shifts,
            split=split, overlap=overlap, num_workers=jobs, segment=segment
        )
        sources = None
        for update in apply_gen:
            if not isinstance(update, torch.Tensor):
                yield update
            else:
                sources = update.squeeze(0)

        if sources is None:
            raise RuntimeError("apply_model_streaming did not yield result tensor")
        
        sources *= ref.std()
        sources += ref.mean()

        kwargs = {
            'samplerate': self.model.samplerate,
            'bitrate': mp3_bitrate,
            'preset': mp3_preset,
            'clip': clip_mode,
            'as_float': wav_output_type == "float32",
            'bits_per_sample': 24 if wav_output_type == "int24" else 16,
        }

        yield self.SeparationStreamProgress.CONVERTING_SOURCES.name
        result = {}
        if stem is None:
            for source, name, i in zip(sources, self.model.sources, range(len(sources))):
                result[name] = convert_audio_wav_to_bytes(source, output_format, **kwargs)
                yield self.SeparationStreamProgress.PROGRESS.value + pickle.dumps({
                    "processed": i + 1,
                    "total": len(sources),
                })
        else:
            sources = list(sources)
            result[stem] = convert_audio_wav_to_bytes(sources.pop(self.model.sources.index(stem)), output_format, **kwargs)
            yield self.SeparationStreamProgress.PROGRESS.value + pickle.dumps({
                "processed": 1,
                "total": 2,
            })
            other_stem = torch.zeros_like(sources[0])
            for i in sources:
                other_stem += i
            result["no_"+stem] = convert_audio_wav_to_bytes(other_stem, output_format, **kwargs)
            yield self.SeparationStreamProgress.PROGRESS.value + pickle.dumps({
                "processed": 2,
                "total": 2,
            })

        yield self.SeparationStreamProgress.END.value + pickle.dumps(result)

    def apply_model_streaming(
        self,
        model: tp.Union[BagOfModels, Model],
        mix: tp.Union[torch.Tensor, TensorChunk],
        shifts: int = 1,
        split: bool = True,
        overlap: float = 0.25,
        transition_power: float = 1.,
        device=None,
        num_workers: int = 0,
        segment: tp.Optional[float] = None,
        pool=None
    ):
        if device is None:
            device = mix.device
        else:
            device = torch.device(device)

        if pool is None:
            if num_workers > 0 and device.type == 'cpu':
                pool = ThreadPoolExecutor(num_workers)
            else:
                pool = DummyPoolExecutor()

        kwargs: tp.Dict[str, tp.Any] = {
            'shifts': shifts,
            'split': split,
            'overlap': overlap,
            'transition_power': transition_power,
            'device': device,
            'pool': pool,
            'segment': segment,
        }

        if isinstance(model, BagOfModels):
            estimates: tp.Union[float, torch.Tensor] = 0.
            totals = [0.] * len(model.sources)
            for i, (sub_model, model_weights) in enumerate(zip(model.models, model.weights)):
                original_model_device = next(iter(sub_model.parameters())).device
                sub_model.to(device)

                for value in self.apply_model_streaming(sub_model, mix, **kwargs):
                    if isinstance(value, torch.Tensor):
                        out = value
                    else:
                        yield value

                sub_model.to(original_model_device)
                for k, inst_weight in enumerate(model_weights):
                    out[:, k, :, :] *= inst_weight
                    totals[k] += inst_weight
                estimates += out
                del out

            assert isinstance(estimates, torch.Tensor)
            for k in range(estimates.shape[1]):
                estimates[:, k, :, :] /= totals[k]
            yield estimates
            return

        model.to(device)
        model.eval()
        assert transition_power >= 1
        batch, channels, length = mix.shape

        if shifts:
            kwargs['shifts'] = 0
            max_shift = int(0.5 * model.samplerate)
            mix = tensor_chunk(mix)
            padded_mix = mix.padded(length + 2 * max_shift)
            out = 0.
            for i in range(shifts):
                offset = random.randint(0, max_shift)
                shifted = TensorChunk(padded_mix, offset, length + max_shift - offset)
                for value in self.apply_model_streaming(model, shifted, **kwargs):
                    if isinstance(value, torch.Tensor):
                        shifted_out = value
                    else:
                        yield value
                out += shifted_out[..., max_shift - offset:]
                yield self.SeparationStreamProgress.PROGRESS.value + pickle.dumps({
                    "processed": i + 1,
                    "total": shifts,
                })
            out /= shifts
            yield out
            return

        if split:
            out = torch.zeros(batch, len(model.sources), channels, length, device=mix.device)
            sum_weight = torch.zeros(length, device=mix.device)
            if segment is None:
                segment = model.segment
            assert segment is not None and segment > 0.
            segment_length = int(model.samplerate * segment)
            stride = int((1 - overlap) * segment_length)
            offsets = list(range(0, length, stride))
            weight = torch.cat([torch.arange(1, segment_length // 2 + 1, device=device),
                            torch.arange(segment_length - segment_length // 2, 0, -1, device=device)])
            weight = (weight / weight.max())**transition_power
            weight = weight.to(mix.device)
            futures = []

            for i, offset in enumerate(offsets):
                chunk = TensorChunk(mix, offset, segment_length)
                future = pool.submit(apply_model, model, chunk, **kwargs)
                futures.append((future, offset))

            total_segments = len(futures)
            start_time = time.time()

            for i, (future, offset) in enumerate(futures):
                chunk_out = future.result()
                chunk_length = chunk_out.shape[-1]
                out[..., offset:offset + segment_length] += (
                    weight[:chunk_length] * chunk_out).to(mix.device)
                sum_weight[offset:offset + segment_length] += weight[:chunk_length].to(mix.device)

                elapsed = time.time() - start_time
                done = i + 1
                eta = (elapsed / done) * (total_segments - done)
                eta_str = time.strftime("%H:%M:%S", time.gmtime(eta))

                yield self.SeparationStreamProgress.PROGRESS.value + pickle.dumps({
                    "processed": done,
                    "total": total_segments,
                    "eta": eta_str
                })

            assert sum_weight.min() > 0
            out /= sum_weight
            yield out
            return

        # Normal non-split, non-shift case
        valid_length: int
        if isinstance(model, HTDemucs) and segment is not None:
            valid_length = int(segment * model.samplerate)
        elif hasattr(model, 'valid_length'):
            valid_length = model.valid_length(length)  # type: ignore
        else:
            valid_length = length
        mix = tensor_chunk(mix)
        padded_mix = mix.padded(valid_length).to(device)
        with torch.no_grad():
            out = model(padded_mix)
        out = center_trim(out, length)
        yield out
