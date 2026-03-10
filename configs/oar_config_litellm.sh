NAME="ann_full_gen_gemini3"
PROJECT_NAME="test1"
HOME="/home/esvirido"
PROJECT_DIR="$HOME/phd/test1"
EMAIL="ekaterina.sviridova@inria.fr"
LOGDIR="$HOME/logs"

# Export API keys
# Google API key
export GOOGLE_API_KEY=$(cat /home/esvirido/.google/api_key)
# OpenAI API key
#export OPENAI_API_KEY=$(cat /home/esvirido/.openai/api_key)

# Make sure the log directory exists
mkdir -p "$LOGDIR"

# No GPU needed for API-based models!
W_HOURS=4                  # Walltime in hours

# Submit the job 
OAR_OUT=$(oarsub \
    --name "$NAME" \
    --directory "$PROJECT_DIR" \
    --stdout="$LOGDIR/%jobid%.stdout" \
    --stderr="$LOGDIR/%jobid%.stderr" \
    --l "nodes=1,walltime=$W_HOURS" \
    --notify "[ERROR,INFO]mail:$EMAIL" \
    "export GOOGLE_API_KEY=$GOOGLE_API_KEY; \
     module load conda; \
     source /home/esvirido/miniconda3/bin/activate /home/esvirido/miniconda3/envs/llm-env; \
     echo 'Starting implicitness reconstruction with gemini...'; \
     python3 scripts/generation/ann_full_gen_gemini.py \
        --input_file data/dialogue/out_dial_json/dev_labeled.jsonl \
        --output_dir results_dev/ann_full_reconstructed_litellm \
        --model 'gemini/gemini-3-flash-preview' \
        --temperature 1.0; \
     echo 'LiteLLM text reconstruction completed.'
    " \
)
   
# Print the job ID / submission output
echo "$OAR_OUT"
