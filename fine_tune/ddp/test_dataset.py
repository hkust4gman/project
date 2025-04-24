from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader 

dataset = load_from_disk('./amazone_dataset')
train_dataloader = DataLoader(dataset['train'], batch_size=32) # more options can be used
val_dataloader = DataLoader(dataset['test'], batch_size=32)
cnt = 0
for batch in train_dataloader:
    if cnt == 1:
        break
    for k, _ in batch.items():
        print(k)
    print(batch['review/score'])
    cnt = cnt + 1
