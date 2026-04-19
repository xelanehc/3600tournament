#!/bin/bash
#SBATCH --job-name=np_test
#SBATCH --output=logs/job_%A_%a.out # logging stuff
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1          
#SBATCH --mem=1G                   
#SBATCH --time=00:20:00            
#SBATCH --array=1-256%64           

cd $SLURM_SUBMIT_DIR/dist

mkdir -p results

VENV_DIR="venv"

module load python 

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in '$VENV_DIR'..."
    python3 -m venv "$VENV_DIR"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create virtual environment."
        exit 1
    fi
else
    echo "Using existing virtual environment '$VENV_DIR'."
fi

echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"
python3 -m pip install --upgrade pip


while IFS="" read -r p || [ -n "$p" ]
do
    echo "Installing $p ..."
    pip install $p
    echo ""
done < requirements.txt

TASK_ID=$SLURM_ARRAY_TASK_ID

echo "Starting task $TASK_ID on $(hostname)"

python3 -u engine/run_local_agents.py Alan5_2 Alan5 > results/output_${TASK_ID}.txt

echo "Task $TASK_ID finished."