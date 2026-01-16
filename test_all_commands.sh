#!/bin/bash

# Test script for all SIMBA commands
# Tests: preprocess, train, inference, analog-discovery

set -e  # Exit on error

echo "================================"
echo "Testing SIMBA CLI Commands"
echo "================================"

# Cleanup previous test runs
rm -rf test_full_workflow/
mkdir -p test_full_workflow

echo ""
echo "1/6 Testing: simba preprocess"
echo "--------------------------------"
uv run simba preprocess \
    preprocessing=fast_dev \
    paths.spectra_path=data/casmi2022.mgf \
    paths.preprocessing_dir=./test_full_workflow/preprocessed/

echo ""
echo "2/6 Testing: simba train"
echo "--------------------------------"
uv run simba train \
    training=fast_dev \
    paths.preprocessing_dir_train=./test_full_workflow/preprocessed/ \
    paths.checkpoint_dir=./test_full_workflow/checkpoints/

echo ""
echo "3/6 Testing: simba inference"
echo "--------------------------------"
uv run simba inference \
    inference=fast_dev \
    --checkpoint-dir ./test_full_workflow/checkpoints/ \
    --preprocessing-dir ./test_full_workflow/preprocessed/ \
    inference.preprocessing_pickle=mapping_unique_smiles.pkl

echo ""
echo "4/6 Testing: simba analog-discovery"
echo "--------------------------------"
uv run simba analog-discovery \
    analog_discovery=fast_dev \
    --model-path ./test_full_workflow/checkpoints/best_model.ckpt \
    --query-spectra data/casmi2022.mgf \
    --reference-spectra data/casmi2022.mgf \
    --output-dir ./test_full_workflow/analog_results/ \
    analog_discovery.query_index=0

echo ""
echo "5/6 Testing: simba inference (pretrained model)"
echo "--------------------------------"
uv run simba inference \
    inference=fast_dev \
    --checkpoint-dir ./downl_data/ \
    --preprocessing-dir ./test_full_workflow/preprocessed/ \
    inference.preprocessing_pickle=mapping_unique_smiles.pkl

echo ""
echo "6/6 Testing: simba analog-discovery (pretrained model)"
echo "--------------------------------"
uv run simba analog-discovery \
    analog_discovery=fast_dev \
    --model-path ./downl_data/best_model.ckpt \
    --query-spectra data/casmi2022.mgf \
    --reference-spectra data/casmi2022.mgf \
    --output-dir ./test_full_workflow/analog_results_pretrained/ \
    analog_discovery.query_index=0

echo ""
echo "7/9 Testing: simba train (with metadata features)"
echo "--------------------------------"
uv run simba train \
    training=fast_dev \
    paths.preprocessing_dir_train=./test_full_workflow/preprocessed/ \
    paths.checkpoint_dir=./test_full_workflow/checkpoints_metadata/ \
    model.features.use_ce=true \
    model.features.use_ion_activation=true \
    model.features.use_ion_method=true

echo ""
echo "8/9 Testing: simba inference (with metadata features)"
echo "--------------------------------"
uv run simba inference \
    inference=fast_dev \
    --checkpoint-dir ./test_full_workflow/checkpoints_metadata/ \
    --preprocessing-dir ./test_full_workflow/preprocessed/ \
    inference.preprocessing_pickle=mapping_unique_smiles.pkl \
    model.features.use_ce=true \
    model.features.use_ion_activation=true \
    model.features.use_ion_method=true

echo ""
echo "9/9 Testing: simba analog-discovery (with metadata features)"
echo "--------------------------------"
uv run simba analog-discovery \
    analog_discovery=fast_dev \
    --model-path ./test_full_workflow/checkpoints_metadata/best_model.ckpt \
    --query-spectra data/casmi2022.mgf \
    --reference-spectra data/casmi2022.mgf \
    --output-dir ./test_full_workflow/analog_results_metadata/ \
    analog_discovery.query_index=0 \
    model.features.use_ce=true \
    model.features.use_ion_activation=true \
    model.features.use_ion_method=true

echo ""
echo "================================"
echo "✓ All commands completed successfully!"
echo "================================"
