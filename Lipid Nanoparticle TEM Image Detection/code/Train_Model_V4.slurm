#!/bin/bash
#SBATCH --job-name=LNPMultiGPU
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4          # [UPDATED] Request 4 GPUs
#SBATCH --cpus-per-task=16    # [UPDATED] Increase CPU cores to feed 4 GPUs
#SBATCH --mem=128G             # [UPDATED] Increase memory for larger batch/multi-gpu
#SBATCH --time=71:59:59
#SBATCH --output=train_multigpu_log_%j.txt
#SBATCH --error=train_multigpu_error_%j.txt

# 1. Clear modules
module purge
module load python

# 2. Activate environment
source /home/jbs263/12_05_2026_YOLOTrainer_1/lnp_env/bin/activate

# 3. Debug info
echo "--- SLURM JOB STARTED ---"
date
echo "Job ID: $SLURM_JOB_ID"
nvidia-smi # Verify all 4 GPUs are visible

# 4. Run the Training Experiments Sequentially
# We use 'python -u' for unbuffered output so logs update in real-time

echo ">>> STARTING RUN 1: 50 EPOCHS"
python -u  Train_Model_V4.py --epochs 50 --name lnp_v5_50_epochs

# 5. End Time
echo "--- SLURM JOB ENDED ---"
date
