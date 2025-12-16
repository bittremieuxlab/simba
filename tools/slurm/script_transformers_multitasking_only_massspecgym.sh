#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH -p ampere_gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1 --cpus-per-task=20

export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate transformers

nvidia-smi



#python training_multitasking_generated_data.py --enable_progress_bar=0    --extra_info=_only_massspecgym  --load_pretrained=1 --pretrained_path=/scratch/antwerpen/209/vsc20939/data/model_checkpoints/pretrained_model_256n/best_model.ckpt --LR=0.0001  --CHECKPOINT_DIR=/scratch/antwerpen/209/vsc20939/data/model_checkpoints/model_256n_only_massspecgym/  --PREPROCESSING_DIR_VAL_TEST=/scratch/antwerpen/209/vsc20939/data/preprocessing_massspecgym_20250415/ --PREPROCESSING_DIR_TRAIN=/scratch/antwerpen/209/vsc20939/data/preprocessing_massspecgym_20250415/  --PREPROCESSING_DIR_VAL_TEST=/scratch/antwerpen/209/vsc20939/data/preprocessing_massspecgym_20250415/ --PREPROCESSING_DIR=/scratch/antwerpen/209/vsc20939/data/preprocessing_massspecgym_20250415/  --PREPROCESSING_PICKLE_FILE=mapping.pkl
python training_multitasking_fingerprint.py --enable_progress_bar=0    --extra_info=_only_massspecgym  --load_pretrained=1 --pretrained_path=/scratch/antwerpen/209/vsc20939/data/model_checkpoints/pretrained_model_256n/best_model.ckpt --LR=0.0001  --CHECKPOINT_DIR=/scratch/antwerpen/209/vsc20939/data/model_checkpoints/model_256n_only_massspecgym_fingerprint/  --PREPROCESSING_DIR_VAL_TEST=/scratch/antwerpen/209/vsc20939/data/preprocessing_massspecgym_20250415/ --PREPROCESSING_DIR_TRAIN=/scratch/antwerpen/209/vsc20939/data/preprocessing_massspecgym_20250415/  --PREPROCESSING_DIR_VAL_TEST=/scratch/antwerpen/209/vsc20939/data/preprocessing_massspecgym_20250415/ --PREPROCESSING_DIR=/scratch/antwerpen/209/vsc20939/data/preprocessing_massspecgym_20250415/  --PREPROCESSING_PICKLE_FILE=mapping.pkl --USE_LOSS_WEIGHTS_SECOND_SIMILARITY=1 --USE_FINGERPRINT=1 --LR=0.0001
