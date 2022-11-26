##
import torch
from torchdata import datapipes as dp
import torchaudio
import os
import numpy as np
import ffmpeg
##
ROOT = './LJSpeech-1.1'
AUDIO_WAV_PATH = "./LJSpeech-1.1/wavs"
SAMPLE_RATE = 22050

##

sample_file = "LJ001-0001.wav"


def load_audio(path):
    waveform, sample_rate = torchaudio.load(path)
    return waveform, sample_rate


audio = load_audio(os.path.join(AUDIO_WAV_PATH, sample_file))

print(audio[0].shape)
##

SAMPLE_RATE = 16000
N_FFT = 400
N_MELS = 80
HOP_LENGTH = 160
CHUNK_LENGTH = 10
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000: number of samples in a chunk


def mel_filters(device, n_mels: int = N_MELS) -> torch.Tensor:
    """
    load the mel filterbank matrix for projecting STFT into a Mel spectrogram.
    Allows decoupling librosa dependency; saved using:
        np.savez_compressed(
            "mel_filters.npz",
            mel_80=librosa.filters.mel(sr=16000, n_fft=400, n_mels=80),
        )
    """
    assert n_mels == 80, f"Unsupported n_mels: {n_mels}"
    with np.load(os.path.join("mel_filters.npz")) as f:
        return torch.from_numpy(f[f"mel_{n_mels}"]).to(device)

def log_mel_spectrogram(audio, n_mels: int = N_MELS):
    """
    Compute the log-Mel spectrogram of
    Parameters
    ----------
    audio: Union[str, np.ndarray, torch.Tensor], shape = (*)
        The path to audio or either a NumPy array or Tensor containing the audio waveform in 16 kHz
    n_mels: int
        The number of Mel-frequency filters, only 80 is supported
    Returns
    -------
    torch.Tensor, shape = (80, n_frames)
        A Tensor that contains the Mel spectrogram
    """
    if not torch.is_tensor(audio):
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio = torch.from_numpy(audio)

    window = torch.hann_window(N_FFT).to(audio.device)
    stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
    magnitudes = stft[:, :-1].abs() ** 2

    filters = mel_filters(audio.device, n_mels)
    print(filters.shape)
    print(magnitudes.shape)
    mel_spec = filters @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec

##
def load_audio_2(file: str, sr: int = SAMPLE_RATE):
    """
    Open an audio file and read as mono waveform, resampling as necessary
    Parameters
    ----------
    file: str
        The audio file to open
    sr: int
        The sample rate to resample the audio if necessary
    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """
    try:
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
##
print(audio[0].squeeze().shape)
spectogram = log_mel_spectrogram(audio[0].squeeze())
print(spectogram.shape)

##
audio_2 = load_audio_2(os.path.join(AUDIO_WAV_PATH, sample_file))
##
print(audio_2.shape)
spectogram = log_mel_spectrogram(audio_2)
print(spectogram.shape)
##


##

