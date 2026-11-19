#!bin/bash

PROJECT_ROOT='/home/vishnubhk/work/jailbreak-evals'
JOB_LOG_DIR=${PROJECT_ROOT}/job_logs/
mkdir -p ${JOB_LOG_DIR}

#SBATCH --account=dt-ta
#SBATCH --partition=TrixieMain
#SBATCH --gres=gpu:1
#SBATCH -c 32
#SBATCH --mem=90G
#SBATCH --job-name=wildjailbreak
#SBATCH --output=${JOB_LOG_DIR}/artifacts_%j.log
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
source /home/vishnubhk/venvs/interp/bin/activate
cd /home/vishnubhk/work/work/jailbreak-evals

python generate.py --model meta-llama/Llama-3.1-8B-Instruct --prompts data/wildjailbreak_train_vanilla.json --output artifacts/wildjailbreak_train_vanilla_responses.jsonl
python generate.py --model meta-llama/Llama-3.1-8B-Instruct --prompts data/wildjailbreak_train_adv.json --output artifacts/wildjailbreak_train_adv_responses.jsonl
python judge.py --prompts data/wildjailbreak_train_vanilla.json --responses artifacts/wildjailbreak_train_vanilla_responses.jsonl --output artifacts/wildjailbreak_train_vanilla_judgments.jsonl
python judge.py --prompts data/wildjailbreak_train_adv.json --responses artifacts/wildjailbreak_train_adv_responses.jsonl --output artifacts/wildjailbreak_train_adv_judgments.jsonl


