#!/bin/bash

set -e
trap "echo '❌ Training interrupted'; exit 1" SIGINT

cd ~/scotland-2026-election-forecast
source .venv/bin/activate

echo "===================================="
echo "TRAINING START: $(date)"
echo "===================================="

# Step 1: Generate data
python3 scripts/generate_data.py

# Step 2: Train models
python3 scripts/train_models.py --n-trials 3

# Ensure model files exist
[ -f models/ensemble.pkl ] || { echo "❌ Missing ensemble.pkl"; exit 1; }
[ -f models/pipeline.pkl ] || { echo "❌ Missing pipeline.pkl"; exit 1; }
[ -f models/shap_importance.csv ] || { echo "❌ Missing shap_importance.csv"; exit 1; }

# Step 3: Create version folder
VERSION=$(date +%Y%m%d_%H%M%S)
VERSION_DIR=models/versions/$VERSION
mkdir -p $VERSION_DIR

# Step 4: Copy model files
cp models/ensemble.pkl $VERSION_DIR/
cp models/pipeline.pkl $VERSION_DIR/
cp models/shap_importance.csv $VERSION_DIR/

echo "📦 Model version stored at: $VERSION_DIR"

# Step 5: Atomic update of latest
mkdir -p models/latest
TMP_DIR=models/latest_tmp
rm -rf $TMP_DIR
mkdir -p $TMP_DIR

cp $VERSION_DIR/* $TMP_DIR/

rm -rf models/latest/*
mv $TMP_DIR/* models/latest/
rmdir $TMP_DIR

echo "🔄 Latest model updated"

echo "===================================="
echo "TRAINING COMPLETE: $VERSION"
echo "===================================="
