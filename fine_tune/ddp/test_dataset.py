from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader, Subset

dataset = load_from_disk('./amazone_dataset')
dataset = dataset.rename_column('review/score', 'labels')
dataset.set_format(type='torch')
dataset = dataset['test']
dataset = Subset(dataset, 100)
dataloader = DataLoader(dataset, batch_size=32) # more options can be used
cnt = 0
for batch in dataloader:
    for label in batch['labels']:
        if label == 1 :
            cnt = cnt + 1
print(cnt)


#from transformers import BertTokenizer
#tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
#local_path = './imdb_dataset'
#dataset = load_dataset("imdb", cache_dir=local_path)
#def tokenize_function(batch):
#    return tokenizer(batch['text'], padding=True, truncation=True)
#
#dataset = dataset.map(tokenize_function, batched=True)
##print(f"rank{config.rank}: dataset column names: {dataset.column_names}")
#dataset = dataset.remove_columns(["text"])
#dataset = dataset.rename_column('label', 'labels')
#dataset.set_format(type='torch')
#dataset.save_to_disk('bert_large')
#train_dataloader = DataLoader(dataset['train'], batch_size=32) # more options can be used
#val_dataloader = DataLoader(dataset['test'], batch_size=32)
#cnt = 0
#for batch in train_dataloader:
#    if cnt == 1:
#        break
#    for k, _ in batch.items():
#        print(k)
#    print(batch['labels'])
#    print(batch['input_ids'][:1])
#    print(batch['attention_mask'][:1])
#    print(batch['token_type_ids'][:1])
#    cnt = cnt + 1