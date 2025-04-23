#!/bin/bash

#SBATCH -J ddp                                      #Slurm job name

#SBATCH -t 48:00:00                                #Maximum runtime of 48 hours

#SBATCH --mail-user=user_name@ust.hk #Update your email address
#SBATCH --mail-type=begin
#SBATCH --mail-type=end

#SBATCH -p gpu


#SBATCH --nodes=1                # node count
#SBATCH --gpus=2                 # number of GPUs per node(only valid under large/normal partition)

#SBATCH -o logs/out%j.log
#SBATCH -e logs/err%j.log


source ~/.bash_profile

cd $HOME//5003/project/fine_tune/ddp
python main.py