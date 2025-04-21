import torch
import util
import os
import torch.distributed as dist
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from torch.utils.data import DataLoader 
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from datasets import load_dataset
from tqdm import tqdm
from torch.optim.lr_scheduler import LinearLR



class Config:
    def __init__(self, device):
        self.device = device
        self.backend = self._get_backend(device)
        self.world_size = self._get_world_size(device)
        self.bert = 'bert-large-uncased'
        self.checkpoint ='./checkpoint.pth' # None if you don't need it.
        self.batch_size = 32
        self.epoch = 10 #TODO: change this
        self.padding_max_length = 512

    def _get_backend(self, device):
        backend = 'gloo' # default option
        if device == 'cpu':
            backend = 'gloo'
        elif device == 'cuda':
            backend = 'nccl'
        return backend
    def _get_world_size(self, device):
        world_size = 8  #default option
        if device == 'cuda':
            world_size = torch.cuda.device_count()
        else:
            world_size = 8
        return world_size
    


def cleanup():
    dist.destroy_process_group()


def main():
    # set this config first please
    device = 'cpu'
    config = Config(device)

    #set up
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    if os.name == 'nt': #windows
        os.environ["GLOO_USE_LIBUV"] = 0

    dist.init_process_group(backend=config.backend, world_size=config.world_size)
    rank = dist.get_rank()

    if config.device == 'cuda':
        torch.cuda.set_device(rank)


    print("loading checkpoint.")
    #loading checkpoint
    checkpoint = None
    try:
        checkpoint=torch.load(config.checkpoint)
    except:
        print('starting with no checkpoint.')

    print("loading model.")
    tokenizer = BertTokenizer.from_pretrained(config.bert)
    model = BertForSequenceClassification.from_pretrained(config.bert, num_labels=2)
    local_model_path = './bert_model'
    model.save_pretrained(local_model_path)

    if config.device == 'cuda':
        model = model.to(rank)
    if checkpoint and rank == 0: # recover the param from checkpoint
        model.load_state_dict(checkpoint['model'])
    model = DDP(model)


    optimizer = AdamW(model.parameters())
    scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.01, total_iters=config.epoch)
    if checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    

    print("loading dataset.")
    local_path = './imdb_dataset'
    dataset = load_dataset("imdb", cache_dir=local_path)
    train_dataset = dataset['train']
    val_dataset = dataset['test']
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=config.padding_max_length, return_tensors='pt')

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)

    sampler = DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, sampler=sampler) # more options can be used

    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size)


    print("training")
    for epoch in range(config.epoch):
        sampler.set_epoch(epoch)
        model.train()

        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch}"):
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
        scheduler.step()
        
        # validation
        dist.barrier() 
        dist.reduce(loss, dst=0)
        if rank == 0:
            avg_loss = loss.item() / config.world_size
            
            raw_model = model.module
            # TODO val_loss to ('cuda') ?
            val_loss = 0
            with torch.no_grad():
                for batch in val_dataloader:
                    inputs, labels = None, None
                    if config.device == 'cuda':
                        inputs = {key: val.to(rank) for key, val in batch.items() if key != 'label'}
                        labels = batch['label'].to(rank)
                    else:
                        inputs = {key: val for key, val in batch.items() if key != 'label'}
                        labels = batch['label']
                    
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