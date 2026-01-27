#!/bin/bash

#SBATCH --job-name=test_3nodes_homo
#SBATCH --partition=one_day
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=32G
#SBATCH --output=logs/test_3nodes_homo_%j.out
#SBATCH --error=logs/test_3nodes_homo_%j.err

# Test 2: 3 homogeneous "nodes" simulated on 1 physical node
# Simulates 3 nodes with same specs: 16 workers each
# Run sequentially as different node processes
# Expected: Data split evenly, results identical to 1-node baseline

mkdir -p logs
cd /home/nkubrakov/simba
source .venv/bin/activate

# Create symlink
ln -sf "MassSpecGym (1).mgf" MassSpecGym.mgf

echo "=== Test 2: 3 Homogeneous Nodes (Simulated on Single Node) ==="
echo "Processing 10k train, 1k val, 1k test with 3 simulated nodes, 16 workers each"
echo "Running on: $(hostname)"

# Clean output directory
rm -rf ./test_results/3nodes_homo
mkdir -p ./test_results/3nodes_homo

# Run node 0
echo "Starting Node 0..."
simba preprocess \
  paths.spectra_path=MassSpecGym.mgf \
  paths.preprocessing_dir=./test_results/3nodes_homo \
  preprocessing.max_spectra_train=10000 \
  preprocessing.max_spectra_val=1000 \
  preprocessing.max_spectra_test=1000 \
  preprocessing.use_lightweight_format=true \
  preprocessing.overwrite=true \
  preprocessing.num_workers=16 \
  preprocessing.num_nodes=3 \
  preprocessing.current_node=0 &
PID0=$!

# Run node 1
echo "Starting Node 1..."
simba preprocess \
  paths.spectra_path=MassSpecGym.mgf \
  paths.preprocessing_dir=./test_results/3nodes_homo \
  preprocessing.max_spectra_train=10000 \
  preprocessing.max_spectra_val=1000 \
  preprocessing.max_spectra_test=1000 \
  preprocessing.use_lightweight_format=true \
  preprocessing.overwrite=true \
  preprocessing.num_workers=16 \
  preprocessing.num_nodes=3 \
  preprocessing.current_node=1 &
PID1=$!

# Run node 2
echo "Starting Node 2..."
simba preprocess \
  paths.spectra_path=MassSpecGym.mgf \
  paths.preprocessing_dir=./test_results/3nodes_homo \
  preprocessing.max_spectra_train=10000 \
  preprocessing.max_spectra_val=1000 \
  preprocessing.max_spectra_test=1000 \
  preprocessing.use_lightweight_format=true \
  preprocessing.overwrite=true \
  preprocessing.num_workers=16 \
  preprocessing.num_nodes=3 \
  preprocessing.current_node=2 &
PID2=$!

# Wait for all to complete
echo "Waiting for all nodes to complete..."
wait $PID0
echo "Node 0 complete (PID $PID0)"
wait $PID1
echo "Node 1 complete (PID $PID1)"
wait $PID2
echo "Node 2 complete (PID $PID2)"

echo "=== Test 2 Complete ==="
echo "Output directory: ./test_results/3nodes_homo"
