import os
import torch
import util
import torch.distributed as dist
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from datasets import load_dataset



class config:
    def __init__(self, device):
        self.device = device
        self.backend = self._get_backend(device)
        self.world_size = self._get_world_size(device)
        self.bert = 'bert-large-uncased'
        self.checkpoint ='checkpoint.pth' # None if you don't need it.
        self.batch_size = 32
        self.epoch = 10 #TODO: change this

    def _get_backend(device):
        backend = 'gloo' # default option
        if device == 'cpu':
            backend = 'gloo'
        elif device == 'cuda':
            backend = 'nccl'
        return backend
    def _get_world_size(device):
        world_size = 4  #default option
        if device == 'cuda':
            world_size = torch.cuda.device_count()
        else:
            world_size = 4
        return world_size
    


def cleanup():
    dist.destroy_process_group()


def main():
    # set this config first please
    device = 'cpu'
    config = config(device)

    #set up
    dist.init_process_group(backend=config.device, world_size=config.world_size)
    rank = dist.get_rank()
    world_size = config.world_size

    if config.device == 'cuda':
        torch.cuda.set_device(rank)


    #loading checkpoint
    checkpoint = None
    try:
        checkpoint=torch.load(config.checkpoint)
    except:
        pass
    

    tokenizer = BertTokenizer.from_pretrained(config.bert)
    model = BertForSequenceClassification.from_pretrained(config.bert, num_labels=2)
    if config.device == 'cuda':
        model = model.to(rank)
    if checkpoint and rank == 0: # recover the param from checkpoint
        model.load_state_dict(checkpoint['model'])
    model = DDP(model)


    #TODO: solve learning rate here!
    optimizer = AdamW(model.parameters(), lr=5e-5)
    if checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    

    #TODO: change dataset here.
    dataset = load_dataset("imdb")
    dataset = dataset['train']
    val_dataset = dataset['test']
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True)

    dataset = dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)

    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, sampler=sampler, shuffle=True) # more options can be used

    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size)


    for epoch in range(config.epoch):
        sampler.set_epoch(config.epoch)
        model.train()

        for batch in dataloader:
            inputs, labels = None, None
            if config.device == 'cuda':
                inputs = {key: val.to(rank) for key, val in batch.items() if key != 'label'}
                labels = batch['label'].to(rank)
            else:
                inputs = {key: val for key, val in batch.items() if key != 'label'}
                labels = batch['label']

                



            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # validation
        dist.reduce(loss, dst=0)
        if rank == 0:
            avg_loss = loss.item() / config.world_size
            
            raw_model = model.module
            val_loss = 0
            with torch.no_grad():
                for input, labels in val_dataloader:
                    inputs = {key: val.squeeze().to(rank) for key, val in inputs.items()}
                    labels = labels.to(rank)
                    
                    outputs = raw_model(**inputs, labels=labels)
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