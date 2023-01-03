##
import torch
import torchdata.datapipes as dp
from torchtext import vocab
from torch.utils.data import DataLoader
from torch.nn.functional import pad
from torch.utils.data.dataset import random_split
from torchdata.datapipes.iter import IterDataPipe, IterableWrapper
import pickle

from collections import Counter

##
ROOT_PATH = "./dataset.pkl"


##
class ParsePickleData(IterDataPipe):
    def __init__(self, source_datapipe):
        self.source_datapipe = source_datapipe
        print("********* caption data **********", self.source_datapipe)

    def __iter__(self):
        for file_name, stream in self.source_datapipe:
            data = pickle.load(stream)
            for item in data:
                yield item['mel_spec'], item['text']


##
def mel_to_tensor(x):
    return torch.from_numpy(x[0]).float(), x[1]


def build_datapipes(path):
    new_dp = IterableWrapper([path])
    new_dp = dp.iter.FileOpener(new_dp, mode='b')
    # https://github.com/pytorch/data/blob/main/examples/text/squad2.py
    # list(new_dp)
    new_dp = ParsePickleData(new_dp)
    new_dp = new_dp.shuffle() \
        .enumerate() \
        .to_map_datapipe() \
        .map(mel_to_tensor)

    return new_dp


##
SAMPLE_RATE = 16000
NUMBER_OF_SAMPLES = 10 * SAMPLE_RATE
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
def collate_batch(batch):
    sos_id = torch.tensor(char_vocab.get_stoi()["<"], dtype=torch.int64)
    eos_id = torch.tensor(char_vocab.get_stoi()[">"], dtype=torch.int64)
    source = []
    target = []

    for i, (x, y) in enumerate(batch):
        text = y
        tokens = char_vocab([t for t in text])
        if (len(tokens) > MAX_TARGET_LENGTH - 2):
            tokens = tokens[:MAX_TARGET_LENGTH - 2]
        tokens = [sos_id] + tokens + [eos_id]
        tokens = torch.tensor(tokens, dtype=torch.int64)
        tokens = torch.nn.functional.pad(tokens, (padding_idx, MAX_TARGET_LENGTH - len(tokens)))
        source.append(x)
        target.append(tokens)
    # print("*********** 1 *******", len(source), len(target))
    return (torch.stack(source), torch.stack(target))


##
##
def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0


##
class Batch:
    def __init__(self, src, target=None, pad=0):  # pad = 2 is the index of the pad that is blank token
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if target is not None:
            self.target = target[:, :-1]
            self.target_y = target[:, 1:]
            self.target_mask = self.make_std_mask(self.target, pad)
            self.ntokens = (self.target_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
        return tgt_mask


##

def create_data_loaders():
    data_iter = build_datapipes(ROOT_PATH)

    def collate_fn(batch):
        return collate_batch(batch)

    train_size = int(0.9 * len(data_iter))
    val_size = len(data_iter) - train_size
    train_dataset, val_dataset = random_split(data_iter, [train_size, val_size])
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader
##


##


##
