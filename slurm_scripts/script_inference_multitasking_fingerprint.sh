#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH -p pascal_gpu
#SBATCH --gpus=1

export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate transformers

nvidia-smi 

#srun  python inference_multitasking_fingerprint.py --enable_progress_bar=0   --CHECKPOINT_DIR=/scratch/antwerpen/209/vsc20939/data/model_checkpoints/model_256n_multitasking_fingerprint_lr00001/ --USE_FINGERPRINT=1
srun  python inference_multitasking_fingerprint.py --enable_progress_bar=0   --CHECKPOINT_DIR=/scratch/antwerpen/209/vsc20939/data/model_checkpoints/model_256n_only_massspecgym_fingerprint/ --enable_progress_bar=0   --USE_FINGERPRINT=1

