##
import torch
import torch.nn as nn
import torch.nn.functional as F
##
from dataset import create_data_loaders, char_vocab
from model import make_model
##
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
asr_model = make_model(char_vocab, N=2, d_model=64, h=2,dropout=0.1, target_max_len=200)
train_loader, val_loader = create_data_loaders()
print(train_loader)
for batch_idx, (mel_grams, target) in enumerate(train_loader):
    images = mel_grams.to(device)
    labels = target.to(device)
    print("images shape", images.shape)
    print("labels shape", labels.shape)
    out = asr_model(mel_grams, target, None, None,)
    # print(out.shape)
    # print(out)
    break
##

