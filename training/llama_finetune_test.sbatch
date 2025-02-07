#!/bin/bash

#SBATCH --job-name=job_name              # Job name
#SBATCH --output=/home/pawangcs/Abhilash/output_%j.txt   # Standard output file
#SBATCH --error=/home/pawangcs/Abhilash/error_%j.txt     # Standard error file
#SBATCH --partition=dgx2                 # Partition name (modify if using a different node
#SBATCH --qos=dgx2                       # QoS (must match the partition)
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks-per-node=1              # Number of tasks per node (max 8)
#SBATCH --gres=gpu:4                     # Number of GPUs (max 8)
#SBATCH --time=72:00:00                  # Max runtime (HH:MM:SS)

# Navigate to the working directory
# cd /raid/pawangcs/

# Load necessary modules
source /scratch/apps/modules/init/bash

module avail

module load cuda/11.7
module load python/3.10

conda info --envs
# Execute the training script
python train_model.py

python3 -V

nvcc --version

nvidia-smi
python llama_finetune_test.py