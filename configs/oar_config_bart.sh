NAME="ann_bart_text_reconstruction"
PROJECT_NAME="test1"
HOME="/home/esvirido"
PROJECT_DIR="$HOME/phd/test1"
EMAIL="ekaterina.sviridova@inria.fr"
LOGDIR="$HOME/logs"
MODEL_ID="facebook/bart-large"
# MODEL_ID="facebook/bart-base"

export HUGGINGFACE_HUB_TOKEN=$(cat /home/esvirido/.huggingface/token)

# Make sure the log directory exists
mkdir -p "$LOGDIR"


W_HOURS=4                  # Walltime in hours
L_NGPUS=1                  # Number of GPUs (BART is small, 1 GPU is sufficient)
P_MINCUDACAPABILITY=6      # Minimum compute capability
P_MINGPUMEMORY=12000       # Minimum GPU memory in MB (12GB is plenty for BART)

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
     source /home/esvirido/miniconda3/bin/activate /home/esvirido/miniconda3/envs/llm-env; \
     echo 'Starting reconstruction with $MODEL_ID...'; \
     python3 scripts/generation/ann_full_gen_bart.py \
        --input_file data/dialogue/out_dial_jsonl/test_labeled.jsonl \
        --output_dir results/ann_full_reconstructed_bart \
        --model_id $MODEL_ID; \
     echo 'Reconstruction with $MODEL_ID completed.'
    " \
)
   
# Print the job ID / submission output
echo "$OAR_OUT"
