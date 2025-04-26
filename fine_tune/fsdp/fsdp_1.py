import os
import random
import torch
import wandb
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset, load_from_disk
from accelerate import FullyShardedDataParallelPlugin
from torch.distributed.fsdp import (
    FullyShardedDataParallel,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from huggingface_hub import login
import torch.distributed as dist
from datetime import timedelta
from torch.distributed import init_process_group
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def setup_wandb(run_name=None):
    wandb.init(
        project="debugging",                                                    
        name=f"fsdp" if run_name is None else run_name
    )

def setup_fsdp_config():
    return {
        "min_num_params": 1e5,  # Add this parameter
        "fsdp_sharding_strategy": 1,  # 1 corresponds to FULL_SHARD
        "fsdp_offload_params": False,
        "fsdp_auto_wrap_policy": "size_based",
        "fsdp_backward_prefetch": "BACKWARD_POST",
        "fsdp_state_dict_type": "FULL_STATE_DICT",
        "fsdp_transformer_layer_cls_to_wrap": "BertLayer",
    }

def cleanup():
    dist.destroy_process_group()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def setup_distributed():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    world_size = dist.get_world_size()
    device_name = f"cuda:{rank}"

    print(f"rank{rank}: Current device: {device_name}")
    return device_name, rank, world_size

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average=None)
    #TODO ? where is your negative precision? f1 recall etc.
    # precision_neg = precision_score(all_labels, all_predictions, pos_label=0, average='binary')
    #do things like this
    acc = accuracy_score(labels, preds)
    
    metrics = {
        'accuracy': acc,
        'f1_0': f1[0],
        'f1_1': f1[1],
        'recall_0': recall[0],
        'recall_1': recall[1],
        'precision_0': precision[0],
        'precision_1': precision[1],
    }
    
    # If running on main process, log to wandb
    if dist.get_rank() == 0:
        wandb.log({
            "train/accuracy": acc,
            "train/f1_negative": f1[0],
            "train/f1_positive": f1[1],
            "train/recall_negative": recall[0],
            "train/recall_positive": recall[1],
            "train/precision_negative": precision[0],
            "train/precision_positive": precision[1],
        })
    
    return metrics

def main():
    debug = True
    torch.cuda.empty_cache()
    _, rank, _ = setup_distributed()

    if rank == 0:
        setup_wandb()

    model = BertForSequenceClassification.from_pretrained(
        "bert-large-uncased",
        num_labels=2,
    )

    batch_count = 4000
    batch_size = 200
    val_num_limit = 1000
    if debug:
        batch_count = 4
        batch_size = 200
        val_num_limit = 100

    try:
        data_path = "../ddp/amazone_dataset"
        tokenized_dataset = load_from_disk(data_path)
        tokenized_dataset = tokenized_dataset.rename_column('review/score', 'labels')
        tokenized_dataset.set_format(type='torch')
        print("Tokenized dataset loaded successfully!")
    except Exception as e:
        print(f"cannot load dataset: {e}")

    train_dataset = Subset(tokenized_dataset['train'], range(batch_count * batch_size))
    eval_dataset = Subset(tokenized_dataset['test'], range(val_num_limit))

    print("Initialize datasets successfully!")

    training_args = TrainingArguments(
        output_dir=f"./bert_finetuned",
        logging_dir=f"./logs",

        learning_rate=1e-5,

        per_device_train_batch_size=batch_size,
        num_train_epochs=1,
        

        # Evaluation settings
        eval_strategy="steps",
        eval_steps= 1 if debug else 50,
        save_strategy="steps",
        save_steps= 1 if debug else 1000,
        
        # fp16=True,

        # Metrics configuration
        metric_for_best_model="eval_loss",
        load_best_model_at_end=True,
        greater_is_better=False,  # For loss, lower is better
        
        # FSDP配置
        fsdp = ["full_shard", "auto_wrap"],
        fsdp_config=setup_fsdp_config(),
        fsdp_transformer_layer_cls_to_wrap="BertLayer",
        
        # 分布式训练配置
        local_rank=rank,
        
        # wandb配置
        #TODO: you don't need to see any info of your servant gpu?
        #report_to="wandb" if rank == 0 else "none",
        report_to="wandb",
        logging_steps=50,
        logging_first_step=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    cleanup()

    if rank == 0:
        wandb.finish()
        
if __name__ == "__main__":
    main()