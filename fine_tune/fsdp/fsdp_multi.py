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

def setup_wandb(run_name=None):
    # 从SLURM环境变量获取作业ID
    job_id = os.environ.get('SLURM_JOB_ID', 'local')
    wandb.init(
        project="bert-finetune",                                                    
        name=f"bert-imdb-fsdp-{job_id}" if run_name is None else run_name,
        config={
            "learning_rate": 2e-5,
            "epochs": 10,
            "batch_size": 32,
            "model_name": "bert-large-uncased",
            "job_id": job_id,
            "num_nodes": int(os.environ.get('SLURM_JOB_NUM_NODES', 1)),
            "num_gpus": int(os.environ.get('SLURM_GPUS_ON_NODE', 1)),
        }
    )

def setup_huggingface_auth():
    """Set up Hugging Face authentication"""
    try:
        # Set up cache directory
        cache_dir = "/home/zsunbl/.cache/huggingface"
        os.makedirs(cache_dir, exist_ok=True)
        
        # Login using token
        login(token="xxx", write_permission=True)
        print("Successfully logged in to Hugging Face")
    except Exception as e:
        print(f"Error during Hugging Face authentication: {e}")
        raise

def setup_fsdp_config():
    return FullyShardedDataParallelPlugin(
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        cpu_offload=CPUOffload(offload_params=False),
    )

def setup_dataset():
    """Helper function to load and set up the IMDB dataset"""
    # Only download/prepare dataset on rank 0
    if dist.get_rank() == 0:
        print("Loading IMDB dataset on rank 0...")
    
    try:
        # Use default cache location and avoid custom data_dir
        dataset = load_dataset(
            "imdb",
            cache_dir="/home/zsunbl/.cache/huggingface/datasets",
            trust_remote_code=True
        )
        if dist.get_rank() == 0:
            print("Dataset loaded successfully!")
        return dataset
    except Exception as e:
        if dist.get_rank() == 0:
            print(f"Error loading dataset: {e}")
        raise

def setup_distributed():
    # 从 Slurm 环境变量获取 rank 和 world_size
    global_rank = int(os.environ["SLURM_PROCID"])
    local_rank = int(os.environ["SLURM_LOCALID"])
    world_size = int(os.environ["SLURM_NTASKS"])

    # 设置主节点地址和端口（Slurm 推荐方式）
    os.environ["MASTER_ADDR"] = os.environ.get("SLURM_LAUNCH_NODE_IPADDR", "localhost")
    os.environ["MASTER_PORT"] = "29500"  # 默认 NCCL 端口
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(global_rank)
    os.environ['LOCAL_RANK'] = str(local_rank)

    # 初始化进程组（NCCL 后端）
    dist.init_process_group(
        backend="nccl",
        rank=global_rank,
        world_size=world_size,
        timeout=timedelta(seconds=1800),  # 避免超时
    )

    # 设置当前 GPU 设备
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    return device, local_rank, global_rank, world_size

def main():
    # Initialize environment variables
    device, local_rank, global_rank, world_size = setup_distributed()

    # 只在主进程初始化wandb
    if global_rank == 0:
        setup_wandb()
    
    # 加载模型和分词器
    model = BertForSequenceClassification.from_pretrained(
        "bert-large-uncased",
        num_labels=2,
    )
    tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")

    class TokenizerWrapper:
        """Wrapper class for tokenizer to handle batched tokenization"""
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer

        def __call__(self, examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=512,
                return_tensors=None  # Ensure we don't get pytorch tensors yet
            )
    
    def setup_dataset(tokenizer):
        """Set up and tokenize the IMDB dataset"""
        if dist.get_rank() == 0:
            print("Loading IMDB dataset...")
        
        try:
            # Set up cache directory
            cache_dir = "/home/zsunbl/.cache/huggingface/datasets"
            os.makedirs(cache_dir, exist_ok=True)
            
            # Load dataset
            dataset = load_dataset(
                "imdb",
                cache_dir=cache_dir,
                trust_remote_code=True
            )
            
            if dist.get_rank() == 0:
                print(f"Dataset loaded. Train samples: {len(dataset['train'])}, Test samples: {len(dataset['test'])}")
            
            # Create tokenizer wrapper
            tokenize_func = TokenizerWrapper(tokenizer)
            
            # Tokenize dataset
            tokenized_dataset = dataset.map(
                tokenize_func,
                batched=True,
                batch_size=1000,
                remove_columns=["text"],
                desc="Tokenizing dataset",
                num_proc=1  # Avoid multiprocessing issues in distributed setting
            )
            
            if dist.get_rank() == 0:
                print("Dataset tokenized successfully!")
                # Print sample to verify tokenization
                sample = tokenized_dataset['train'][0]
                print(f"Sample tokenized output length: {len(sample['input_ids'])}")
            
            return tokenized_dataset
        
        except Exception as e:
            if dist.get_rank() == 0:
                print(f"Error in dataset setup: {str(e)}")
            raise


    #加载数据集
    cache_path = "tokenized_dataset"
    if os.path.exists(cache_path) and dist.get_rank() == 0:
        print("Loading cached tokenized dataset...")
        try:
            tokenized_dataset = load_from_disk(cache_path)
            print("Cached dataset loaded successfully!")
        except Exception as e:
            print(f"Error loading cached dataset: {e}")
            os.remove(cache_path)
            tokenized_dataset = setup_dataset(tokenizer)
    else:
        tokenized_dataset = setup_dataset(tokenizer)
        if dist.get_rank() == 0:
            tokenized_dataset.save_to_disk(cache_path)
            print("Tokenized dataset saved to disk.")
    
    #加载小数据集：
    # dataset = load_dataset("imdb")
    # dataset["train"] = dataset["train"].select(range(50))
    # dataset["test"] = dataset["test"].select(range(20))

    # def tokenization_function(examples):
    #     return tokenizer(
    #         examples["text"],
    #         truncation=True,
    #         padding="max_length",
    #         max_length=512,
    #         return_tensors=None
    #     )

    # tokenized_dataset = dataset.map(tokenization_function,batched=True)


    # 修改训练参数
    training_args = TrainingArguments(
        output_dir=f"./bert_finetuned_{os.environ.get('SLURM_JOB_ID', 'local')}",
        per_device_train_batch_size=64,
        num_train_epochs=10,
        logging_dir=f"./logs_{os.environ.get('SLURM_JOB_ID', 'local')}",
        save_steps=500,
        eval_steps=500,
        fp16=True,
        
        # FSDP配置
        fsdp="full_shard auto_wrap",
        fsdp_transformer_layer_cls_to_wrap="BertLayer",
        
        # 分布式训练配置
        local_rank=local_rank,
        
        # wandb配置
        report_to="wandb" if global_rank == 0 else "none",
        logging_steps=50,
        logging_first_step=True,
    )

    model = model.to(device)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
    )
    
    trainer.train()
    
    if global_rank == 0:
        wandb.finish()

if __name__ == "__main__":
    main()