#!/bin/bash

#SBATCH --job-name=test_1node
#SBATCH --partition=one_day
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=logs/test_1node_%j.out
#SBATCH --error=logs/test_1node_%j.err

# Test 1: Single node (baseline)
# Expected: All data processed by one node

mkdir -p logs
cd /home/nkubrakov/simba
source .venv/bin/activate

# Create symlink
ln -sf "MassSpecGym (1).mgf" MassSpecGym.mgf

echo "=== Test 1: Single Node Baseline ==="
echo "Processing 10k train, 1k val, 1k test with 1 node, 16 workers"

simba preprocess \
  paths.spectra_path=MassSpecGym.mgf \
  paths.preprocessing_dir=./test_results/1node \
  preprocessing.max_spectra_train=10000 \
  preprocessing.max_spectra_val=1000 \
  preprocessing.max_spectra_test=1000 \
  preprocessing.use_lightweight_format=true \
  preprocessing.overwrite=true \
  preprocessing.num_workers=16 \
  preprocessing.num_nodes=1 \
  preprocessing.current_node=0

echo "=== Test 1 Complete ==="
echo "Output directory: ./test_results/1node"
