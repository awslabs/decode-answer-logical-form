#!/bin/bash

conda env config vars set DATA_DIR=$1/data
conda env config vars set SAVE_DIR=$1/outputs
conda env config vars set MODEL_DIR=$1/models

echo "DATA_DIR: $1/data"
echo "SAVE_DIR: $1/outputs"
echo "MODEL_DIR: $1/models"

# Create directories
mkdir -p $1/data
mkdir -p $1/outputs
mkdir -p $1/models