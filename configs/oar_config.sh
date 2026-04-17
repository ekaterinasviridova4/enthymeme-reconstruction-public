NAME="reconstruction_job"
PROJECT_NAME="your_project"
HOME="/home/your_user"
PROJECT_DIR="$HOME/path/to/project"
EMAIL="your.email@example.com"
LOGDIR="$HOME/logs"
#MODEL_ID="mistralai/Mistral-Small-3.2-24B-Instruct-2506"
MODEL_ID="allenai/Olmo-3.1-32B-Instruct"
export HUGGINGFACE_HUB_TOKEN=$(cat /your/key/here)

# Make sure the log directory exists
mkdir -p "$LOGDIR"


W_HOURS=4                  # Walltime in hours
L_NGPUS=4                  # Number of GPUs
P_MINCUDACAPABILITY=7      # Minimum compute capability
P_MINGPUMEMORY=24000       # Minimum GPU memory in MB

# Submit the job
OAR_OUT=$(oarsub \
    --name "$NAME" \
    --directory "$PROJECT_DIR" \
    --stdout="$LOGDIR/%jobid%.stdout" \
    --stderr="$LOGDIR/%jobid%.stderr" \
    --property="gpu_compute_capability>='$P_MINCUDACAPABILITY' and gpu_mem>='$P_MINGPUMEMORY'" \
    --l "nodes=1/gpu=$L_NGPUS,walltime=$W_HOURS" \
    --notify "[ERROR,INFO]mail:$EMAIL" \
    "export HUGGINGFACE_HUB_TOKEN=$HUGGINGFACE_HUB_TOKEN; \
     module load conda; \
     source $HOME/miniconda3/bin/activate your-env; \
     echo 'Starting reconstruction with $MODEL_ID...'; \
     python3 scripts/generation/ann_short_gen_mistral24.py \
        --input_file path/to/your/input.jsonl \
        --output_dir path/to/your/output_dir \
        --model_id $MODEL_ID; \
     echo 'Reconstruction with $MODEL_ID completed.'
    " \
)
    #--stdout=logs/%jobid%.stdout \
    #--stderr=logs/%jobid%.stderr \
   
# Print the job ID / submission output
echo "$OAR_OUT"

