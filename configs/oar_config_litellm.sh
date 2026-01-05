NAME="cot_speaker_exchanges_gpt4o"
PROJECT_NAME="test1"
HOME="/home/esvirido"
PROJECT_DIR="$HOME/phd/test1"
EMAIL="ekaterina.sviridova@inria.fr"
LOGDIR="$HOME/logs"

# Export API keys
# Google API key
#export GOOGLE_API_KEY=$(cat /home/esvirido/.google/api_key)
# OpenAI API key
export OPENAI_API_KEY=$(cat /home/esvirido/.openai/api_key)

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
    "export OPENAI_API_KEY=$OPENAI_API_KEY; \
     module load conda; \
     source /home/esvirido/miniconda3/bin/activate /home/esvirido/miniconda3/envs/llm-env; \
     echo 'Starting CoT speaker exchange reconstruction with GPT-4o...'; \
     python3 scripts/generation/cot_full_gen_gemini.py \
        --input_file data/processed/speaker_exchanges_dev.jsonl \
        --output_dir results/cot_reconstructed_litellm \
        --model 'openai/gpt-4o' \
        --limit 5 \
        --temperature 0.0; \
     echo 'LiteLLM text reconstruction completed.'
    " \
)
   
# Print the job ID / submission output
echo "$OAR_OUT"
