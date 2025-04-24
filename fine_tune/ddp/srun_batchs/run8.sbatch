#!/bin/bash

#SBATCH -J ddp8                                      #Slurm job name

#SBATCH -t 2:00:00                                #Maximum runtime of 48 hours

#SBATCH --mail-user=srongaa@ust.hk #Update your email address
#SBATCH --mail-type=begin
#SBATCH --mail-type=end

#SBATCH -p normal 


#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=8      # total number of tasks per node
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gpus=8                 # number of GPUs per node(only valid under large/normal partition)
#SBATCH --account=mscbdt2024

#SBATCH -o logs/out%j.log
#SBATCH -e logs/err%j.log

export MASTER_PORT=12348
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

source ~/.bashrc

cd $HOME//5003/project/fine_tune/ddp
torchrun --nproc_per_node=8 --rdzv_endpoint=localhost:12348 main.py