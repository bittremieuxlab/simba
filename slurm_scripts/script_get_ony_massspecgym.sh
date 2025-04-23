#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH -p zen2
#SBATCH --ntasks=1 --cpus-per-task=20

export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate transformers

nvidia-smi 

## Run gumbel
#srun python training_multitasking_generated_data.py --EDIT_DISTANCE_USE_GUMBEL=1 --enable_progress_bar=0  --D_MODEL=256  --ADD_HIGH_SIMILARITY_PAIRS=0 --extra_info=_generated_data_gumbel  --load_pretrained=1 --LR=0.00001

## Running 10 layers
#srun python training_multitasking_generated_data.py --enable_progress_bar=0  --N_LAYERS=10 --D_MODEL=128  --ADD_HIGH_SIMILARITY_PAIRS=0 --extra_info=_generated_data_20250206_no_pretraining  --load_pretrained=0 --LR=0.00001


## Running 5 layers
#srun python training_multitasking_generated_data.py --enable_progress_bar=0    --extra_info=_smooth_penalty_matrix  --load_pretrained=1 --pretrained_path=/scratch/antwerpen/209/vsc20939/data/model_checkpoints/pretrained_model_256n/best_model.ckpt --LR=0.00001


python get_only_massspecgym.py 