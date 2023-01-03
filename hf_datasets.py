##
from datasets import load_dataset
##
voxpopuli_croatian = load_dataset("/Users/pasha/Desktop/startup/dubsub", "en",split="train", streaming=True)
##
#%%
print(next(iter(voxpopuli_croatian)))
##
print(next(iter(voxpopuli_croatian)))

##

