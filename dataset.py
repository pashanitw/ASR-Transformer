##
from torchdata import datapipes as dp
import torchaudio
import os
import torch
from torchtext import vocab
from collections import Counter
from torch.utils.data import DataLoader, random_split
##
ROOT = './LJSpeech-1.1'
AUDIO_WAV_PATH = "./LJSpeech-1.1/wavs"
SAMPLE_RATE = 22050
NUMBER_OF_SAMPLES = 10*SAMPLE_RATE
MAX_TARGET_LENGTH = 200
##
padding_idx = 0
chars = (
            ["-", "#", "<", ">"]
            + [chr(i + 96) for i in range(1, 27)]
            + [" ", ".", ",", "?"]
        )

char_vocab = vocab.vocab(Counter(chars))
char_vocab.set_default_index(char_vocab["#"])
print(char_vocab(["3"]))
##

AUDIO_TRANSFORMS = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=512, n_mels=64
)
##
def load_audio(path):
    waveform, _ = torchaudio.load(path)
    waveform = waveform.mean(0)
    waveform = waveform[:NUMBER_OF_SAMPLES]
    waveform = waveform / waveform.abs().max()
    return waveform
##
def load_text(path):
    with open(path, "r") as f:
        text = f.read()
    return text
##
def filter_csv(x):
    return x.endswith('.csv')

def text_split_fn(column):
    return (os.path.join(AUDIO_WAV_PATH, column[0]+".wav"), column[2])

def load_librispeech_item(data):
    audio_file, transcript = data
    waveform, sample_rate = torchaudio.load(audio_file)
    return (
        waveform,
        transcript,
        sample_rate,
    )

def resample_waveform(x):
    sample_rate = x[2]
    waveform = x[0]
    if sample_rate != SAMPLE_RATE:
        waveform = torchaudio.transforms.Resample(
            sample_rate, SAMPLE_RATE
        )(waveform)
    return (waveform, x[1], SAMPLE_RATE)

def mix_down(x):
    waveform, transcript, sample_rate = x
    if waveform.shape[0] > 1:
        waveform = waveform.mean(0)
    return (waveform, transcript, sample_rate)

def cut_waveform(x):
    waveform, transcript, sample_rate = x
    if waveform.shape[1] > NUMBER_OF_SAMPLES:
        waveform = waveform[:, :NUMBER_OF_SAMPLES]
    return (waveform, transcript, sample_rate)

def right_pad_waveform(x):
    waveform, transcript, sample_rate = x
    if waveform.shape[1] < NUMBER_OF_SAMPLES:
        waveform = torch.nn.functional.pad(
            waveform, (padding_idx, NUMBER_OF_SAMPLES - waveform.shape[1])
        )
    return (waveform, transcript)

def apply_transforms(x):
    waveform, transcript = x
    return (AUDIO_TRANSFORMS(waveform), transcript)


def get_librispeech_data():
    data = (
        dp.iter.LoadFilesFromDisk([AUDIO_WAV_PATH])
        .filter(filter_csv)
        .map(load_text)
        .map(text_split_fn)
        .map(load_librispeech_item)
        .map(resample_waveform)
        .map(mix_down)
        .map(cut_waveform)
        .map(right_pad_waveform)
        .map(apply_transforms)
        .map(text_to_tensor)
    )
    return data
##
def build_datapipes(path):
    new_dp = dp.iter.FileLister([path])\
                    .filter(filter_csv)

    new_dp = new_dp.open_files(mode="rt")
    new_dp = new_dp.parse_csv(delimiter='|')
    new_dp = new_dp.shuffle()
    # new_dp = new_dp.map(text_split_fn)
    return new_dp.enumerate()\
            .to_map_datapipe() \
            .map(text_split_fn) \
            .map(load_librispeech_item) \
            .map(resample_waveform) \
            .map(mix_down) \
            .map(cut_waveform) \
            .map(right_pad_waveform) \
            .map(apply_transforms)

##
def collate_batch(batch):
    sos_id =torch.tensor(char_vocab.get_stoi()["<"], dtype=torch.int64)
    eos_id = torch.tensor(char_vocab.get_stoi()[">"], dtype=torch.int64)
    source = []
    target = []

    for i, (x, y) in enumerate(batch):
        text = y
        tokens = char_vocab([t for t in text])
        if(len(tokens) > MAX_TARGET_LENGTH-2):
            tokens = tokens[:MAX_TARGET_LENGTH-2]
        tokens = [sos_id] + tokens + [eos_id]
        tokens = torch.tensor(tokens, dtype=torch.int64)
        tokens = torch.nn.functional.pad(tokens, (padding_idx, MAX_TARGET_LENGTH - len(tokens)))
        source.append(x)
        target.append(tokens)
    print("*********** 1 *******", len(source), len(target))
    return (torch.stack(source), torch.stack(target))
##
def create_data_loaders():
    data_iter = build_datapipes(ROOT)

    def collate_fn(batch):
        return collate_batch(batch)

    train_size = int(0.8 * len(data_iter))
    val_size = len(data_iter) - train_size
    print("************", train_size, val_size)
    train_dataset, val_dataset = random_split(data_iter, [train_size, val_size])
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader
##

