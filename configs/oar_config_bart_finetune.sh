NAME="bart_finetune_test"
PROJECT_NAME="test1"
HOME="/home/esvirido"
PROJECT_DIR="$HOME/phd/test1"
EMAIL="ekaterina.sviridova@inria.fr"
LOGDIR="$HOME/logs"
MODEL_ID="allenai/led-large-16384"

export HUGGINGFACE_HUB_TOKEN=$(cat /home/esvirido/.huggingface/token)

# Make sure the log directory exists
mkdir -p "$LOGDIR"


W_HOURS=4                  # Walltime in hours
L_NGPUS=4                  # Number of GPUs
P_MINCUDACAPABILITY=6      # Minimum compute capability
P_MINGPUMEMORY=32000       # Minimum GPU memory in MB

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
    source ~/.bashrc; \
    conda activate llm-env; \
    echo 'Starting Fine-tuning...'; \
    python scripts/generation/finetune_bart.py \
    --train_file data/dialogue/out_dial_jsonl/post_processed_train.jsonl \
    --limit_train_data 20 \
    --output_dir results/bart_finetuned/model_bart_large \
    --model_name $MODEL_ID \
    --epochs 10 \
    --batch_size 1 \
    --learning_rate 2e-5 \
    --max_length 4096; \
    echo 'Starting Inference on Test Set...'; \
    python scripts/generation/ann_finetuned_gen_bart.py \
    --test_file data/dialogue/out_dial_jsonl/post_processed_test.jsonl \
    --model_path results/bart_finetuned/model_bart_large \
    --output_file results/ann_full_reconstructed_bart/finetuned_reconstructed_test_bart.jsonl \
    --batch_size 1 \
    --max_length 4096")
    
echo "Submitted job: $OAR_OUT"
