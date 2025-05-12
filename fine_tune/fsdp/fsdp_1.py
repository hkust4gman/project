import os
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
    # 从SLURM环境变量获取作业ID
    job_id = os.environ.get('SLURM_JOB_ID', 'local')
    wandb.init(
        project="bertft-fsdp-4gpu-b64*2",                                                    
        name=f"fsdp-{job_id}" if run_name is None else run_name
    )

def setup_fsdp_config():
    return FullyShardedDataParallelPlugin(
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        cpu_offload=CPUOffload(offload_params=False),
    )

def cleanup():
    # 清理分布式环境
    dist.destroy_process_group()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def setup_distributed():
    n_gpu = torch.cuda.device_count()
    if n_gpu == 0:
        raise RuntimeError("No CUDA GPUs available")

    # 从 Slurm 环境变量获取 rank 和 world_size
    global_rank = int(os.environ["SLURM_PROCID"])
    local_rank = int(os.environ["SLURM_LOCALID"])
    world_size = int(os.environ["SLURM_NTASKS"])

    if global_rank == 0:
        print(f"Available GPUs: {n_gpu}")
        for i in range(n_gpu):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    # Verify GPU assignment
    if local_rank >= n_gpu:
        raise RuntimeError(f"Invalid local_rank: {local_rank}. Only {n_gpu} GPUs available")

    # 设置主节点信息
    os.environ["MASTER_ADDR"] = os.environ.get("SLURM_LAUNCH_NODE_IPADDR", "localhost")
    os.environ["MASTER_PORT"] = "12334"  # 使用固定端口
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(global_rank)
    os.environ['LOCAL_RANK'] = str(local_rank)

    
    dist.init_process_group(
        backend="nccl",
        world_size=world_size,
        rank=global_rank,
        timeout=timedelta(seconds=1800),
    )

    # 设置当前GPU
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        raise RuntimeError("CUDA is not available")

    if global_rank == 0:
        print(f"World size: {world_size}, Number of available GPUs: {n_gpu}")
    print(f"Process {global_rank} using GPU {local_rank} ({torch.cuda.get_device_name(local_rank)})")

    return device, local_rank, global_rank, world_size

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average=None)
    acc = accuracy_score(labels, preds)
    
    # Create metrics dictionary
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
    # if dist.get_rank() == 0:
    #     wandb.log({
    #         "train/accuracy": acc,
    #         "train/f1_negative": f1[0],
    #         "train/f1_positive": f1[1],
    #         "train/recall_negative": recall[0],
    #         "train/recall_positive": recall[1],
    #         "train/precision_negative": precision[0],
    #         "train/precision_positive": precision[1],
    #     })
    
    #单卡
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
    torch.cuda.empty_cache()
    # Initialize environment variables
    #device, local_rank, global_rank, world_size = setup_distributed()

    # 只在主进程初始化wandb
    # if global_rank == 0:
    #     setup_wandb()

    # 单卡训练
    setup_wandb()

    # 加载模型和分词器
    model = BertForSequenceClassification.from_pretrained(
        "bert-large-uncased",
        num_labels=2,
    )

    #单卡训练
    model = model.to("cuda")

    batch_count = 4000
    batch_size = 128
    val_num_limit = 1000

    #加载数据集
    try:
        data_path = "./amazone_dataset/amazone_dataset"
        tokenized_dataset = load_from_disk(data_path)
        tokenized_dataset = tokenized_dataset.rename_column('review/score', 'labels')
        tokenized_dataset.set_format(type='torch')
        print("Tokenized dataset loaded successfully!")
    except Exception as e:
        print(f"cannot load dataset: {e}")

    train_dataset = Subset(tokenized_dataset['train'], range(batch_count * batch_size))
    eval_dataset = Subset(tokenized_dataset['test'], range(val_num_limit))

    print("Initialize datasets successfully!")

    # 修改训练参数
    training_args = TrainingArguments(
        output_dir=f"./bert_finetuned_{os.environ.get('SLURM_JOB_ID', 'local')}",
        logging_dir=f"./logs_{os.environ.get('SLURM_JOB_ID', 'local')}",

        learning_rate=1e-5,

        per_device_train_batch_size=64,
        dataloader_drop_last=True,
        gradient_accumulation_steps=2,
        num_train_epochs=1,
        

        # Evaluation settings
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=1000,
        # fp16=True,

        # Metrics configuration
        metric_for_best_model="eval_loss",
        load_best_model_at_end=True,
        greater_is_better=False,
        
        # # FSDP配置
        # fsdp = "full_shard auto_wrap",
        # fsdp_transformer_layer_cls_to_wrap="BertLayer",
        
        # 分布式训练配置
        #local_rank=local_rank,
        
        # wandb配置
        #report_to="wandb" if global_rank == 0 else "none",
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

    # if global_rank == 0:
    #     wandb.finish()
    wandb.finish()
        
if __name__ == "__main__":
    main()