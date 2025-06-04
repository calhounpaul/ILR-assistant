#!/bin/bash
# Configuration
ENV_NAME="typical_tune"
PYTHON_VERSION="3.11"

TRAIN_BINARY_COMMAND="venv/bin/accelerate launch"
#SCRIPT_PATH="tune_only.py"
SCRIPT_PATH="7_finetune.py"

# Essential NCCL timeout and communication settings
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1

# Network interface specification (most critical fix)
export NCCL_SOCKET_IFNAME=enX0  # Replace with your actual interface
export GLOO_SOCKET_IFNAME=enX0
export NCCL_SOCKET_FAMILY=AF_INET

# Disable P2P if causing issues
export NCCL_P2P_DISABLE=1

# Fix tokenizer parallelism conflicts
export TOKENIZERS_PARALLELISM=false

#TRAIN_BINARY_COMMAND="python3"
#SCRIPT_PATH="tune.py"

# Check if the conda environment exists
if conda info --envs | grep -q "^$ENV_NAME "; then
  echo "Environment '$ENV_NAME' already exists, activating..."
else
  echo "Environment '$ENV_NAME' does not exist, creating..."
  conda create -y -n $ENV_NAME python=$PYTHON_VERSION

  echo "Installing required packages..."
  conda activate $ENV_NAME
  pip install -U datasets transformers torch diffusers accelerate bitsandbytes numpy trl peft
  conda deactivate

  echo "Environment setup complete!"
fi

# Run the script within the conda environment
#echo "Running $SCRIPT_PATH in the '$ENV_NAME' environment..."
#source "$(conda info --base)/etc/profile.d/conda.sh"
#conda activate $ENV_NAME

HF_TOKEN=$(cat hf_token.txt) \
  $TRAIN_BINARY_COMMAND $SCRIPT_PATH

conda deactivate
echo "Execution complete!"
