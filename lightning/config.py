import torch

# Set device cuda for GPU if it's available otherwise run on the CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training hyperparameters
INPUT_SIZE = 784
NUM_CLASSES = 10
LEARNING_RATE = 0.001
BATCH_SIZE = 64
NUM_EPOCHS = 1
LEARNING_RATE = 0.001

# Dataset
DATA_DIR = "dataset/"
NUM_WORKERS = 8

# Compute
ACCELERATOR="cpu"
PRECISION=16