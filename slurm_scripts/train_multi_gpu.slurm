#!/bin/bash

# --- SLURM Directives ---
#SBATCH --job-name=c3net_train_8gpu   # Descriptive job name
#SBATCH --partition=gpu               # CUSTOMIZE: Your cluster's GPU partition name
#SBATCH --nodes=1                     # Request 1 node
#SBATCH --ntasks=1                    # Usually 1 task for distributed training
#SBATCH --cpus-per-task=16            # CUSTOMIZE: CPUs per GPU (e.g., 8 CPUs × 2 = 16 for 2 GPUs)
#SBATCH --mem=64G                     # CUSTOMIZE: RAM based on your needs
#SBATCH --time=24:00:00               # CUSTOMIZE: Max wall time (format: HH:MM:SS)

# --- GPU Request ---
#SBATCH --gres=gpu:8                  # CUSTOMIZE: Number of GPUs to request

# --- Output Files ---
# CUSTOMIZE: Update paths to your project location
#SBATCH --output=logs/slurm_train_%j.out
#SBATCH --error=logs/slurm_train_%j.err

# --- Environment Setup ---
echo "------------------------------------------------------"
echo "Job Started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node List: $SLURM_JOB_NODELIST"
echo "Partition: $SLURM_JOB_PARTITION"
echo "GPU Request: $SLURM_GPUS_ON_NODE GPUs"
echo "------------------------------------------------------"

echo "Activating Conda environment..."
# CUSTOMIZE: Update environment name if different
source activate c3net
echo "Conda environment activated: $CONDA_DEFAULT_ENV"

# --- Project Setup ---
# CUSTOMIZE: Update to your C3Net project path
PROJECT_ROOT="$HOME/C3Net"  # Example: Change to your actual path
cd $PROJECT_ROOT
echo "Current directory: $(pwd)"

# Create logs directory if it doesn't exist
mkdir -p logs

echo "Starting multi-GPU training..."
# --- Run Training ---
python main.py train

echo "------------------------------------------------------"
echo "Training script finished."
echo "Attempting to start evaluation..."

# --- Evaluation Step ---
# Define the base directory for run outputs, relative to PROJECT_ROOT
RUNS_BASE_DIR_REL="results/training/runs"
FULL_RUNS_BASE_PATH="${PROJECT_ROOT}/${RUNS_BASE_DIR_REL}"

LATEST_RUN_FULL_PATH=""
MODEL_PATH_ARG=""

# Check if the base directory for runs exists
if [ ! -d "$FULL_RUNS_BASE_PATH" ]; then
    echo "Error: Runs directory not found: $FULL_RUNS_BASE_PATH"
    echo "Evaluation cannot proceed."
else
    # Find the latest directory starting with run_ or runs_ within FULL_RUNS_BASE_PATH
    # -t sorts by modification time (newest first)
    # -d lists directories themselves, rather than their contents
    # The path pattern ensures we search within the correct base directory
    # 2>/dev/null suppresses errors if no matching directories are found
    LATEST_RUN_FULL_PATH=$(ls -td ${FULL_RUNS_BASE_PATH}/run_*/ ${FULL_RUNS_BASE_PATH}/runs_*/ 2>/dev/null | head -n 1)

    if [ -z "$LATEST_RUN_FULL_PATH" ]; then
        echo "Error: Could not find any run directory (e.g., run_YYYYMMDD_HHMMSS/ or runs_YYYYMMDD_HHMMSS/) in $FULL_RUNS_BASE_PATH"
        echo "Please check the path and naming convention of training output directories."
        echo "Evaluation cannot proceed."
    else
        # LATEST_RUN_FULL_PATH will be an absolute path like /path/to/project/results/training/runs/run_XXXXXX/ (with a trailing slash)
        echo "Latest run directory found: $LATEST_RUN_FULL_PATH"

        # Construct the model path argument relative to PROJECT_ROOT for python main.py
        # This strips the PROJECT_ROOT prefix from LATEST_RUN_FULL_PATH.
        # Example: LATEST_RUN_FULL_PATH = /hdd3/SLURM/home/slurm_baber/C3Net_L_392_E23_Final_Guidance_Different_Loss/results/training/runs/run_XYZ/
        #          PROJECT_ROOT         = /hdd3/SLURM/home/slurm_baber/C3Net_L_392_E23_Final_Guidance_Different_Loss
        # Resulting RELATIVE_RUN_PATH   = results/training/runs/run_XYZ/
        RELATIVE_RUN_PATH=${LATEST_RUN_FULL_PATH#${PROJECT_ROOT}/}
        
        # Ensure RELATIVE_RUN_PATH does not start with a slash if PROJECT_ROOT was "/" (unlikely here but good practice)
        if [[ "$PROJECT_ROOT" == "/" && "${RELATIVE_RUN_PATH:0:1}" == "/" ]]; then
            RELATIVE_RUN_PATH=${RELATIVE_RUN_PATH#/}
        fi

        MODEL_PATH_ARG="${RELATIVE_RUN_PATH}checkpoints/model_best.pth"
        
        # The full absolute path to the checkpoint file for existence check
        FULL_MODEL_CHECKPOINT_PATH="${LATEST_RUN_FULL_PATH}checkpoints/model_best.pth"

        echo "Constructed model path for evaluation (relative to project root): $MODEL_PATH_ARG"

        # Check if the model file actually exists
        if [ -f "$FULL_MODEL_CHECKPOINT_PATH" ]; then
            echo "Starting evaluation with model: $MODEL_PATH_ARG..."
            # Ensure we are in PROJECT_ROOT when running main.py (already set)
            python main.py evaluate --model "$MODEL_PATH_ARG"
            echo "Evaluation script finished."
        else
            echo "Error: Model checkpoint not found at $FULL_MODEL_CHECKPOINT_PATH"
            echo "Please check if 'model_best.pth' was created in the 'checkpoints' subdirectory of the latest run ($LATEST_RUN_FULL_PATH)."
            echo "Evaluation cannot proceed."
        fi
    fi
fi

echo "------------------------------------------------------"
echo "Job Ended: $(date)"
echo "------------------------------------------------------"