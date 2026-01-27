#!/bin/bash

#SBATCH --job-name=test_3nodes_hetero
#SBATCH --partition=one_day
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=32G
#SBATCH --output=logs/test_3nodes_hetero_%j.out
#SBATCH --error=logs/test_3nodes_hetero_%j.err

# Test 3: 3 heterogeneous "nodes" simulated on 1 physical node
# Simulates Janne's thermo servers with different worker counts
# Node 0: 20 workers (like 64-core server)
# Node 1: 10 workers (like 32-core server)
# Node 2: 15 workers (like 48-core server)
# Expected: Data split evenly, different processing speeds

mkdir -p logs
cd /home/nkubrakov/simba
source .venv/bin/activate

# Create symlink
ln -sf "MassSpecGym (1).mgf" MassSpecGym.mgf

echo "=== Test 3: 3 Heterogeneous Nodes (Simulated Thermo Servers) ==="
echo "Processing 10k train, 1k val, 1k test with 3 simulated nodes, different worker counts"
echo "Running on: $(hostname)"

# Clean output directory
rm -rf ./test_results/3nodes_hetero
mkdir -p ./test_results/3nodes_hetero

# Run node 0 with 20 workers
echo "Starting Node 0 with 20 workers (simulating 64-core server)..."
simba preprocess \
  paths.spectra_path=MassSpecGym.mgf \
  paths.preprocessing_dir=./test_results/3nodes_hetero \
  preprocessing.max_spectra_train=10000 \
  preprocessing.max_spectra_val=1000 \
  preprocessing.max_spectra_test=1000 \
  preprocessing.use_lightweight_format=true \
  preprocessing.overwrite=true \
  preprocessing.num_workers=20 \
  preprocessing.num_nodes=3 \
  preprocessing.current_node=0 &
PID0=$!

# Run node 1 with 10 workers
echo "Starting Node 1 with 10 workers (simulating 32-core server)..."
simba preprocess \
  paths.spectra_path=MassSpecGym.mgf \
  paths.preprocessing_dir=./test_results/3nodes_hetero \
  preprocessing.max_spectra_train=10000 \
  preprocessing.max_spectra_val=1000 \
  preprocessing.max_spectra_test=1000 \
  preprocessing.use_lightweight_format=true \
  preprocessing.overwrite=true \
  preprocessing.num_workers=10 \
  preprocessing.num_nodes=3 \
  preprocessing.current_node=1 &
PID1=$!

# Run node 2 with 15 workers
echo "Starting Node 2 with 15 workers (simulating 48-core server)..."
simba preprocess \
  paths.spectra_path=MassSpecGym.mgf \
  paths.preprocessing_dir=./test_results/3nodes_hetero \
  preprocessing.max_spectra_train=10000 \
  preprocessing.max_spectra_val=1000 \
  preprocessing.max_spectra_test=1000 \
  preprocessing.use_lightweight_format=true \
  preprocessing.overwrite=true \
  preprocessing.num_workers=15 \
  preprocessing.num_nodes=3 \
  preprocessing.current_node=2 &
PID2=$!

# Wait for all to complete
echo "Waiting for all nodes to complete..."
wait $PID0
echo "Node 0 complete (PID $PID0) - 20 workers"
wait $PID1
echo "Node 1 complete (PID $PID1) - 10 workers"
wait $PID2
echo "Node 2 complete (PID $PID2) - 15 workers"

echo "=== Test 3 Complete ==="
echo "Output directory: ./test_results/3nodes_hetero"
