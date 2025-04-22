import torch
import util
import os
import torch.distributed as dist
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from torch.utils.data import DataLoader 
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from datasets import load_dataset, load_from_disk
from tqdm import tqdm



class Config:
    def __init__(self, device):
        self.device = device
        self.backend = self._get_backend(device)
        self.world_size = -1 
        self.bert = 'bert-large-uncased'
        self.checkpoint ='./checkpoint.pth' # None if you don't need it.
        self.batch_size = 32
        self.epoch = 10 #TODO: change this
        self.padding_max_length = 512
        self.multi_node = False
        self.rank = -1
        self.dataset_batch_size = 1000
        self.lr = 0.001


    def _get_backend(self, device):
        backend = 'gloo' # default option
        if device == 'cpu':
            backend = 'gloo'
        elif device == 'cuda':
            backend = 'nccl'
        return backend
    


def cleanup():
    dist.destroy_process_group()


def main():
    # set this config first please
    print("initializing")
    device = 'cpu'
    config = Config(device)

    if device == 'cpu' and not config.multi_node:
        dist.init_process_group(backend=config.backend)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        config.world_size = world_size
        config.rank = rank 

    if config.device == 'cuda':
        #TODO: do the cuda things
        torch.cuda.set_device(rank)


    print(f"rank{config.rank}: loading checkpoint.")
    #loading checkpoint
    checkpoint = None
    try:
        checkpoint=torch.load(config.checkpoint)
    except:
        print(f"rank{config.rank}: starting with no checkpoint.")

    print(f"rank{config.rank}: loading model.")
    tokenizer = BertTokenizer.from_pretrained(config.bert)
    model = BertForSequenceClassification.from_pretrained(config.bert, num_labels=2)
    local_model_path = './bert_model'
    model.save_pretrained(local_model_path)

    if config.device == 'cuda':
        model = model.to(config.rank)
    if checkpoint and config.rank == 0: # recover the param from checkpoint
        model.load_state_dict(checkpoint['model'])
    model = DDP(model)


    optimizer = AdamW(model.parameters(), lr=config.lr)
    if checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    

    print(f"rank{config.rank}: loading dataset.")
    dataset = None 
    try:
        dataset = load_from_disk('tokenized_dataset')
    except:
        print(f"rank{config.rank}: loading not tokenized dataset.")
        local_path = './imdb_dataset'
        dataset = load_dataset("imdb", cache_dir=local_path)
        def tokenize_function(batch):
            return tokenizer(batch['text'], padding=True, truncation=True)

        dataset = dataset.map(tokenize_function, batched=True)
        #print(f"rank{config.rank}: dataset column names: {dataset.column_names}")
        dataset = dataset.remove_columns(["text"])
        dataset = dataset.rename_column('label', 'labels')
        dataset.set_format(type='torch')
        dataset.save_to_disk('tokenized_dataset')
        #print(f"rank{config.rank}: dataset column names: {dataset.column_names}")

    sampler = DistributedSampler(dataset['train'])
    train_dataloader = DataLoader(dataset['train'], batch_size=config.batch_size, sampler=sampler) # more options can be used
    val_dataloader = DataLoader(dataset['test'], batch_size=config.batch_size)
   # for batch in train_dataloader:
   #     print(f'rank{config.rank}:')
   #     for k, v in batch.items():
   #         print(k, v)
   #     break


    print(f"rank{config.rank}: training")
    for epoch in range(config.epoch):
        sampler.set_epoch(epoch)
        model.train()

        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch}"):
            inputs = batch
            if config.device == 'cuda':
                inputs = {k: v.to(config.rank) for k, v in inputs.items()}
            else:
                inputs = {k: v for k, v in inputs.items()}


            print(f"rank{config.rank}: getting outputs")
            #DEBUG: some bug here, program crash when getting outputs
            try:
                outputs = model(**inputs)
                print(f"rank{config.rank}: getting loss")
                loss = outputs.loss
                print(f"rank{config.rank}: loss:{loss}")
                optimizer.zero_grad()
                loss.backward()
                print(f"rank{config.rank}: backward")
                optimizer.step()
            except Exception as e:
                print(f'Error occurred: {e}')
        
        # validation
        dist.barrier() 
        dist.reduce(loss, dst=0)
        if config.rank == 0:
            avg_loss = loss.item() / config.world_size
            
            raw_model = model.module
            # TODO val_loss to ('cuda') ?
            val_loss = 0
            with torch.no_grad():
                for batch in val_dataloader:
                    inputs = batch
                    if config.device == 'cuda':
                        inputs = {k: v.to(config.rank) for k, v in inputs.items()}
                    else:
                        inputs = {k: v for k, v in inputs.items()}
                    
                    outputs = raw_model(**inputs)
                    loss = outputs.loss
                    val_loss += loss.item()

            val_avg_loss = val_loss / len(val_dataloader)
            print(f'train_loss:{avg_loss,}, val_loss:{val_avg_loss}')
            checkpoint_filename = util.generate_filename_with_timestamp('checkpoint', 'pth')
            torch.save({'model':model.module.state_dict(), 'optimizer':optimizer.state_dict()}, checkpoint_filename)
        
        dist.barrier() 


    cleanup()


if __name__ == "__main__":
    main()