#!/bin/bash
#SBATCH --job-name=ImgGenBck	
#SBATCH --output=ImgGenBck_log_%j.txt
#SBATCH --error=ImgGenBck_error_%j.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00        # Increased time for 2000 images + 50 epochs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

module purge
module load python

# Activate your existing environment
source /home/jbs263/12_05_2026_YOLOTrainer_1/lnp_env/bin/activate

# Run the python script
python generateimages_wbckgrdimg.py
