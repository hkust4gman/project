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
import wandb




class Config:
    def __init__(self, device):
        self.device = device
        self.backend = self._get_backend(device)
        dist.init_process_group(backend=self.backend)
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.bert = 'bert-large-uncased'
        self.bert_save_path = self._get_bert_save_path(self.bert)
        self.dataset_save_path = self._get_dataset_save_path(self.bert)
        self.checkpoint ='none' # None if you don't need it.
        self.batch_size = 32 
        self.epoch = 2 #TODO: change this
        self.padding_max_length = 512
        self.multi_node = False
        self.lr = 1e-5
        self.device_name = self._get_device_name(self.device, self.rank)
        self.debug = False 
        if self.debug:
            self.batch_size = 1
            self.epoch = 2
            self.bert = 'bert-base-uncased'

    def _get_device_name(self, device, rank):
        return f"{device}:{rank}"
    def _get_bert_save_path(self, bert_type):
        if bert_type == 'bert-large-uncased':
            return 'bert-large-uncased'
        if bert_type == 'bert-base-uncased':
            return 'bert-base-uncased'

    def _get_dataset_save_path(self, bert_type):
        if bert_type == 'bert-large-uncased':
            return 'dataset_large'
        if bert_type == 'bert-base-uncased':
            return 'dataset_base'

    def _get_backend(self, device):
        backend = 'gloo' # default option
        if device == 'cpu':
            backend = 'gloo'
        elif device == 'cuda':
            backend = 'nccl'
        return backend

    def print_variables(self):
        for var_name, var_value in self.__dict__.items():
            print(f"{var_name}: {var_value}")
    


def cleanup():
    dist.destroy_process_group()


def main():
    # set this config first please
    print("initializing")
    device = 'cuda'
    config = Config(device)
    config.print_variables()

    if config.rank == 0:
        project_name = util.generate_filename_with_timestamp(f"{config.bert}_{config.batch_size}_{config.device}_{config.lr}_{config.world_size}", '')
        print("initializing wandb")
        wandb.init(project=project_name)
        print("finished initializing wandb")

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
    local_model_path = config.bert_save_path
    tokenizer.save_pretrained(local_model_path)
    model.save_pretrained(local_model_path)

    model = model.to(config.device_name)
    if checkpoint and config.rank == 0: # recover the param from checkpoint
        model.load_state_dict(checkpoint['model'])
    model = DDP(model)


    optimizer = AdamW(model.parameters(), lr=config.lr)
    if checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    

    print(f"rank{config.rank}: loading dataset.")
    dataset = None 
    try:
        dataset = load_from_disk(config.dataset_save_path)
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
        dataset.save_to_disk(config.dataset_save_path)
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
        train_dataloader = list(train_dataloader)[:10] if config.debug else train_dataloader
        cnt = 0
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch}"):
            inputs = batch
            inputs = {k: v.to(config.device_name) for k, v in inputs.items()}


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
                max_allocated_memory, max_reserved_memory, allocated_memory, reserved_memory = None, None, None, None
                if torch.cuda.is_available():
                    max_allocated_memory = torch.cuda.max_memory_allocated()
                    max_reserved_memory = torch.cuda.max_memory_reserved()
                    print(f"Max allocated memory: {max_allocated_memory / 1024**2} MB")
                    print(f"Max reserved memory: {max_reserved_memory / 1024**2} MB")
                    allocated_memory = torch.cuda.memory_allocated()
                    reserved_memory = torch.cuda.memory_reserved()
                    print(f"Allocated memory: {allocated_memory / 1024**2} MB")
                    print(f"Reserved memory: {reserved_memory / 1024**2} MB")
                wandb.log({
                    'batch cnt': cnt,
                    "train loss": loss,
                    "alloc cuda memo": allocated_memory,
                    "reserved cuda memo": reserved_memory,
                    "max_alloc cuda memo": max_allocated_memory,
                    "max_reserved cuda memo": max_reserved_memory
                })
            except Exception as e:
                print(f'Error occurred: {e}')
            cnt = cnt + 1
        
        # validation
        dist.barrier() 
        dist.reduce(loss, dst=0)
        if config.rank == 0:
            avg_loss = loss.item() / config.world_size
            
            raw_model = model.module
            # TODO val_loss to ('cuda') ?
            val_loss = 0
            with torch.no_grad():
                val_dataloader = list(val_dataloader)[:10] if config.debug else val_dataloader
                for batch in tqdm(val_dataloader, desc=f"Epoch {epoch}"):
                    inputs = batch
                    inputs = {k: v.to(config.device_name) for k, v in inputs.items()}
                    
                    outputs = raw_model(**inputs)
                    loss = outputs.loss
                    val_loss += loss.item()

            val_avg_loss = val_loss / len(val_dataloader)
            print(f'train_loss:{avg_loss,}, val_loss:{val_avg_loss}')

            wandb.log({
                "val loss": avg_loss,
                "train avg loss": val_avg_loss,
                "epoch": epoch,
            })
            checkpoint_filename = util.generate_filename_with_timestamp(f'checkpoint_{config.bert}', 'pth')
            torch.save({'model':model.module.state_dict(), 'optimizer':optimizer.state_dict()}, checkpoint_filename)
        
        dist.barrier() 

    wandb.finish()
    cleanup()


if __name__ == "__main__":
    main()