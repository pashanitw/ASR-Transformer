##
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
##
from dataset import create_data_loaders, char_vocab, Batch
from model import make_model
from optimizer import get_optmimizer
##
# check normalized
# check padding
# check sos and eos

init_lr = 0.00001
lr_after_warmup = 0.001
final_lr = 0.00001
warmup_epochs = 15
decay_epochs = 85

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
asr_model = make_model(char_vocab, N=2,n_mels=80, d_model=64, h=2,dropout=0.1, target_max_len=200)
asr_model = asr_model.to(device)
module = asr_model
print(module.generator)
train_loader, val_loader = create_data_loaders()
no_steps_per_epoch = len(train_loader)
print("*******************", no_steps_per_epoch)
optimizer, lr_scheduler = get_optmimizer(asr_model, no_steps_per_epoch, init_lr, lr_after_warmup, warmup_epochs, decay_epochs, final_lr)

for batch_idx, (mel_grams, target) in enumerate(train_loader):
    optimizer.zero_grad()
    mel_grams = mel_grams.to(device)
    target = target.to(device)
    batch = Batch(mel_grams, target, pad=0)
    print("audio shape", mel_grams.shape)
    print("labels shape", target.shape)
    out = asr_model.forward(batch.src, batch.target, None, batch.target_mask)
    out = module.generator(out)
    print("out shape", out.shape, batch.target_y.shape)
    # print("===== output shapes =====")
    # print(out.view(-1, out.size(-1)).shape, batch.target_y.contiguous().view(-1).shape)
    loss = F.cross_entropy(out.view(-1, out.size(-1)), batch.target_y.contiguous().view(-1))
    print("loss", loss)
    loss.backward()
    optimizer.step()
    lr_scheduler.step()
    if batch_idx % 10 == 0:
         print("batch_idx", batch_idx, "loss", loss.item())

    if batch_idx == 10:
        break
    # print(out)
    break
##


##


##

