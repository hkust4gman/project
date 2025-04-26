import torch
import time
import util
import os
import torch.distributed as dist
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data.distributed import DistributedSampler
from torch.amp import GradScaler, autocast
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
import math
import wandb




class Config:
    def __init__(self, device):
        self.device = device
        self.backend = self._get_backend(device)
        dist.init_process_group(backend=self.backend)
        self.rank = dist.get_rank()
        if self.device == 'cuda':
            torch.cuda.set_device(self.rank)
        self.world_size = dist.get_world_size()
        self.bert = 'bert-large-uncased'
        self.bert_save_path = self._get_bert_save_path(self.bert)
        self.amazon = True
        self.fp16 = False
        if self.amazon:
            self.dataset_save_path = "../ddp/amazone_dataset"
        else:
            self.dataset_save_path = self._get_dataset_save_path(self.bert)
        self.checkpoint ='none' # None if you don't need it.
        self.batch_size = 200 
        self.epoch = 1 
        self.padding_max_length = 512
        self.multi_node = False
        self.lr = 1e-5
        self.device_name = self._get_device_name(self.device, self.rank)
        batch_count = 4000
        self.num_limit = batch_count * self.batch_size
        self.eval_interval_per_x_batch = math.ceil(batch_count / self.world_size / 10)
        self.val_num_limit = 1000
        self.debug = False 
        if self.debug:
            self.batch_size = 200 
            self.epoch = 1
            batch_count = 100
            self.num_limit = batch_count * self.batch_size
            self.eval_interval_per_x_batch = batch_count // self.world_size // 10 # i think there should be 10 eval.
            self.val_num_limit = 100

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
    
    def set_fp16(self):
        self.fp16 = True
        return self

    def set_debug(self):
        self.debug = True
        return self

    def print_variables(self):
        for var_name, var_value in self.__dict__.items():
            print(f"{var_name}: {var_value}")
    


def cleanup():
    dist.destroy_process_group()


def main():
    # set this config first please
    device = 'cuda'
    config = Config(device)
    config = config.set_fp16()
    config.print_variables()

    project_name = util.generate_filename_with_timestamp(f"fsdp_{config.bert}_{config.batch_size}_{config.device}_{config.lr}_{config.world_size}_{config.rank}", '')
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
    model = FSDP(model)


    optimizer = AdamW(model.parameters(), lr=config.lr)
    if checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    scaler = GradScaler()

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
                if torch.cuda.is_available():
                    current_device = torch.cuda.current_device()
                    print(f"rank{config.rank}: Current device: {current_device}")
                    torch.cuda.synchronize(current_device)
                    pre_forward_allocated = torch.cuda.memory_allocated(current_device) / 1024**3
                    pre_forward_reserved = torch.cuda.memory_reserved(current_device) / 1024**3
                    print(f"rank{config.rank}: pre_forward_allocated: {pre_forward_allocated} GB")
                    print(f"rank{config.rank}: pre_forward_reserved: {pre_forward_reserved} GB")
                    wandb.log({
                        "pre_forward_allocated": pre_forward_allocated,
                        "pre_forward_reserved": pre_forward_reserved
                    })
    
                loss = None
                if config.fp16:
                    with autocast(config.device):
                        outputs = model(**inputs)
                        print(f"rank{config.rank}: getting loss")
                        loss = outputs.loss
                        print(f"rank{config.rank}: loss:{loss}")
                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    print(f"rank{config.rank}: backward")
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(**inputs)
                    print(f"rank{config.rank}: getting loss")
                    loss = outputs.loss
                    print(f"rank{config.rank}: loss:{loss}")
                    optimizer.zero_grad()
                    loss.backward()
                    print(f"rank{config.rank}: backward")
                    optimizer.step()
                if torch.cuda.is_available():
                    torch.cuda.synchronize(current_device)
                    post_backward_allocated = torch.cuda.memory_allocated(current_device) / 1024**3
                    post_backward_reserved = torch.cuda.memory_reserved(current_device) / 1024**3
                    max_allocated = torch.cuda.max_memory_allocated(current_device) / 1024**3
                    print(f"rank{config.rank}: post_backward_allocated: {post_backward_allocated} GB")
                    print(f"rank{config.rank}: post_backward_reserved: {post_backward_reserved} GB")
                    print(f"rank{config.rank}: max_allocated: {max_allocated} GB")
                    wandb.log({
                        "post_backward_allocated": post_backward_allocated,
                        "post_backward_reserved": post_backward_reserved,
                        "max_allocated": max_allocated
                    })
                wandb.log({
                    'batch cnt': i,
                    "train loss": loss,
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
                    precision_neg = precision_score(all_labels, all_predictions, pos_label=0, average='binary')
                    precision_pos = precision_score(all_labels, all_predictions, pos_label=1, average='binary')
                    recall_neg = recall_score(all_labels, all_predictions, pos_label=0, average='binary')
                    recall_pos = recall_score(all_labels, all_predictions, pos_label=1, average='binary')
                    f1_neg = f1_score(all_labels, all_predictions, pos_label=0, average='binary')
                    f1_pos = f1_score(all_labels, all_predictions, pos_label=1, average='binary')
                    acc = correct / total
                    print(f'rank{config.rank}:train_loss:{avg_loss,}, val_loss:{val_avg_loss}, acc:{acc:.4f}, prec_neg{precision_neg}, recall_neg:{recall_neg}, f1_neg:{f1_neg}, prec_pos{precision_pos}, recall_pos:{recall_pos}, f1_pos:{f1_pos}')

                    wandb.log({
                        "val loss": avg_loss,
                        "train avg loss": val_avg_loss,
                        "acc": acc,
                        "eval index": ((i + 1) // eval_interval),
                        "epoch": epoch,
                        "duration": duration,
                        "precision0": precision_neg,
                        "precision1": precision_pos,
                        "recall0": recall_neg,
                        "recall1": recall_pos,
                        "f1_0": f1_neg,
                        "f1_1": f1_pos
                    })
                    checkpoint_filename = util.generate_filename_with_timestamp(f'checkpoint_{config.bert}', 'pth')
                    print(f'checkpoint saved as {checkpoint_filename}.')
                    torch.save({'model':model.module.state_dict(), 'optimizer':optimizer.state_dict()}, checkpoint_filename)
        
        dist.barrier() 

    wandb.finish()
    cleanup()


if __name__ == "__main__":
    main()