import torch
import time
import util
import os
import torch.distributed as dist
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
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
        self.amazon = True
        if self.amazon:
            self.dataset_save_path = "amazone_dataset"
        else:
            self.dataset_save_path = self._get_dataset_save_path(self.bert)
        self.checkpoint ='none' # None if you don't need it.
        self.batch_size = 200 
        self.epoch = 1 
        self.padding_max_length = 512
        self.multi_node = False
        self.lr = 1e-5
        self.device_name = self._get_device_name(self.device, self.rank)
        batch_count = 22500
        self.num_limit = batch_count* self.batch_size
        self.eval_interval_per_x_batch = batch_count // 10
        self.val_num_limit = 1000
        self.debug = False 
        if self.debug:
            self.batch_size = 10
            self.epoch = 2
            batch_count = 100 
            self.num_limit = batch_count* self.batch_size
            self.eval_interval_per_x_batch = batch_count // 10
            self.val_num_limit = 10

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
    device = 'cuda'
    config = Config(device)
    config.print_variables()

    project_name = util.generate_filename_with_timestamp(f"{config.bert}_{config.batch_size}_{config.device}_{config.lr}_{config.world_size}_{config.rank}", '')
    wandb.init(project=project_name)

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
        dataset.set_format(type='torch')
        dataset = dataset.rename_column('review/score', 'labels')
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

    train_dataset = dataset['train']
    train_dataset = Subset(train_dataset, range(config.num_limit))
    sampler = DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, sampler=sampler) # more options can be used
    val_dataset = dataset['test']
    val_dataset = Subset(val_dataset, range(config.val_num_limit))
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size)
   # for batch in train_dataloader:
   #     print(f'rank{config.rank}:')
   #     for k, v in batch.items():
   #         print(k, v)
   #     break


    print(f"rank{config.rank}: training")
    for epoch in range(config.epoch):
        sampler.set_epoch(epoch)
        model.train()
        start_time = time.time()  
        eval_interval = config.eval_interval_per_x_batch
        for i, batch in enumerate(train_dataloader):
            inputs = batch
            inputs = {k: v.to(config.device_name) for k, v in inputs.items()}


            print(f"rank{config.rank}: getting outputs")
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
                    max_allocated_memory = torch.cuda.max_memory_allocated() / 1024**3
                    max_reserved_memory = torch.cuda.max_memory_reserved() / 1024**3
                    print(f"rank{config.rank}: Max allocated memory: {max_allocated_memory} GB")
                    print(f"rank{config.rank}: Max reserved memory: {max_reserved_memory} GB")
                    allocated_memory = torch.cuda.memory_allocated() / 1024**3
                    reserved_memory = torch.cuda.memory_reserved() / 1024**3
                    print(f"rank{config.rank}: allocated memory: {allocated_memory} GB")
                    print(f"rank{config.rank}: Reserved memory: {reserved_memory} GB")
                wandb.log({
                    'batch cnt': i,
                    "train loss": loss,
                    "alloc cuda memo": allocated_memory,
                    "reserved cuda memo": reserved_memory,
                    "max_alloc cuda memo": max_allocated_memory,
                    "max_reserved cuda memo": max_reserved_memory
                })
            except Exception as e:
                print(f'Error occurred: {e}')
            
            if (i + 1) % eval_interval == 0:
                end_time = time.time() 
                duration = end_time - start_time 
                print(f"eval {(i + 1) // eval_interval}: finished in {duration / 60**2}h.", duration)
                # validation
                dist.barrier() 
                dist.reduce(loss, dst=0)
                if config.rank == 0:
                    avg_loss = loss.item() / config.world_size
                    
                    raw_model = model.module
                    val_loss = 0
                    correct = 0
                    total =0
                    all_predictions = []
                    all_labels = []
                    with torch.no_grad():
                        for j, batch in enumerate(val_dataloader):
                            inputs = batch
                            inputs = {k: v.to(config.device_name) for k, v in inputs.items()}
                            
                            outputs = raw_model(**inputs)
                            loss = outputs.loss
                            val_loss += loss.item()

                            #acc
                            predictions = torch.argmax(outputs.logits, dim=-1)
                            all_predictions.extend(predictions.cpu().numpy())
                            all_labels.extend(inputs['labels'].cpu().numpy())

                            correct += (predictions == inputs['labels']).sum().item()
                            total += inputs['labels'].size(0)

                    val_avg_loss = val_loss / len(val_dataloader)
                    precision = precision_score(all_labels, all_predictions, average='weighted')
                    recall = recall_score(all_labels, all_predictions, average='weighted')
                    f1 = f1_score(all_labels, all_predictions, average='weighted')
                    acc = correct / total
                    print(f'rank{config.rank}:train_loss:{avg_loss,}, val_loss:{val_avg_loss}, acc:{acc:.4f}, prec{precision:.4f}, recall:{recall:.4f}, f1:{f1:.4f},')

                    wandb.log({
                        "val loss": avg_loss,
                        "train avg loss": val_avg_loss,
                        "acc": acc,
                        "prec": precision,
                        "recall": recall,
                        "f1": f1,
                        "eval index": ((i + 1) // eval_interval),
                        "epoch": epoch,
                        "duration": duration
                    })
                    checkpoint_filename = util.generate_filename_with_timestamp(f'checkpoint_{config.bert}', 'pth')
                    torch.save({'model':model.module.state_dict(), 'optimizer':optimizer.state_dict()}, checkpoint_filename)
        
        dist.barrier() 

    wandb.finish()
    cleanup()


if __name__ == "__main__":
    main()