##
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchinfo import summary
##
from data_loder import create_data_loaders, char_vocab, Batch
from model import make_model
from optimizer import get_optmimizer

##
# check normalized
# check padding
# check sos and eos
##
init_lr = 0.00001
lr_after_warmup = 0.001
final_lr = 0.00001
warmup_epochs = 80
decay_epochs = 700
##
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
asr_model = make_model(char_vocab, N=2, n_mels=80, d_model=64, h=2, dropout=0.1, target_max_len=200)
asr_model = asr_model.to(device)
module = asr_model
##
#print(summary(asr_model, [(1,80,1000),(1,199),(1,1000,1000), (1,199,199)], dtypes=[torch.float, torch.long, torch.long, torch.long], depth=10))
##
train_loader, val_loader = create_data_loaders()
no_steps_per_epoch = len(train_loader)
optimizer, lr_scheduler = get_optmimizer(asr_model, no_steps_per_epoch, init_lr, lr_after_warmup, warmup_epochs,
                                         decay_epochs, final_lr)
##
summary_printed = False
epochs = 800
accum_iter = 5
for epoch in range(epochs):
    asr_model.train()
    total_traning_loss = 0
    steps = 0
    for batch_idx, (mel_grams, target) in enumerate(train_loader):
        optimizer.zero_grad()
        mel_grams = mel_grams.to(device)
        target = target.to(device)
        batch = Batch(mel_grams, target, pad=0)
        # print("audio shape", mel_grams.shape)
        if summary_printed == False:
            print(summary(asr_model, input_data=[batch.src,batch.target,torch.randint(0,1,(batch.src.shape[0],batch.src.shape[2],batch.src.shape[2])),batch.target_mask], depth=10))
            summary_printed = True

        out = asr_model.forward(batch.src, batch.target, None, batch.target_mask)
        out = module.generator(out)
        print("output shape", out.shape)
        # print("out shape", out.shape, batch.target_y.shape)
        # print("===== output shapes =====")
        # print(out.view(-1, out.size(-1)).shape, batch.target_y.contiguous().view(-1).shape)
        loss = F.cross_entropy(out.view(-1, out.size(-1)), batch.target_y.contiguous().view(-1), label_smoothing=0.1)
        # print("loss", loss)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        total_traning_loss += loss.item()
        steps += 1
        if batch_idx % 40 == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(
                (
                        "Epoch Step: %6d  | Loss: %6.2f "
                        + " | Learning Rate: %6.1e"
                )
                % (batch_idx, total_traning_loss / steps, lr)
            )

    torch.save(asr_model.state_dict(), f"./{epoch}-asr_model.pt")
    asr_model.eval()
    total_loss = 0
    for batch_idx, (mel_grams, target) in enumerate(val_loader):
        mel_grams = mel_grams.to(device)
        target = target.to(device)
        batch = Batch(mel_grams, target, pad=0)
        out = asr_model.forward(batch.src, batch.target, None, batch.target_mask)
        out = module.generator(out)
        loss = F.cross_entropy(out.view(-1, out.size(-1)), batch.target_y.contiguous().view(-1), label_smoothing=0.1)
        total_loss += loss.item()
    print("epoch ", epoch, "val_loss", total_loss / len(val_loader))

##
