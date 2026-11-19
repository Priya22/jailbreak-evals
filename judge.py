import argparse
import json
from pathlib import Path
import numpy as np
from vllm import LLM, SamplingParams

def load_prompts_and_repsonses(prompts_path, responses_path):
    '''
    prompts_path: .json file with "prompt_id" and "prompt" keys
    responses_path: .json file with "prompt_id" and "response" keys
    returns: list of prompts and list of responses matched by prompt_id
    '''
    prompts_text = Path(prompts_path).read_text(encoding="utf-8")
    if prompts_path.endswith(".json"):
        prompts_records = json.loads(prompts_text)
    else:
        prompts_records = [json.loads(line) for line in prompts_text.splitlines() if line.strip()]

    responses_text = Path(responses_path).read_text(encoding="utf-8")
    if responses_path.endswith(".json"):
        responses_records = json.loads(responses_text)
    else:
        responses_records = [json.loads(line) for line in responses_text.splitlines() if line.strip()]

    prompts_dict = {r["prompt_id"]: r["prompt"] for r in prompts_records}
    records = []
    for r in responses_records:
        if "prompt_id" not in r or "response" not in r:
            raise ValueError(f"Record must have 'prompt_id' and 'response' keys")
        prompt_id = r["prompt_id"]
        if prompt_id not in prompts_dict:
            raise ValueError(f"Response with prompt_id {prompt_id} has no matching prompt")
        records.append({"prompt_id": prompt_id, "prompt": prompts_dict[prompt_id], "response": r["response"]})
    
    return records


def score_with_llama_guard(prompts, responses):

    sampling_params = SamplingParams(temperature= 0, max_tokens= 100)
    model_name = 'meta-llama/Llama-Guard-3-8B'

    llm = LLM(model_name, max_model_len=4096, gpu_memory_utilization=0.6)

    msgs = [[{"role": "user", "content": p}, {"role":"assistant", "content": r}] for p,r in zip(prompts, responses)]

    ops = llm.chat(msgs, sampling_params=sampling_params)

    scores = []
    outputs = []

    for op in ops:
        txt = op.outputs[0].text.strip()
        prts = txt.split("\n")
        score = np.nan
        if prts[0].strip().lower() == 'safe':
            score = 0
        elif prts[0].strip().lower() == 'unsafe':
            score = 1
        scores.append(score)
        outputs.append(txt.strip())

    return scores, outputs

def main():
    parser = argparse.ArgumentParser(description="Judge responses using LLaMA-Guard")
    parser.add_argument("--prompts", required=True, help="Path to prompts file (.json or .jsonl)")
    parser.add_argument("--responses", required=True, help="Path to responses file (.json or .jsonl)")
    parser.add_argument("--output", required=True, help="Path to output JSONL file with scores and outputs")
    args = parser.parse_args()

    records = load_prompts_and_repsonses(args.prompts, args.responses)
    prompts = [r["prompt"] for r in records]
    responses = [r["response"] for r in records]

    scores, outputs = score_with_llama_guard(prompts, responses)

    with open(args.output, "w", encoding="utf-8") as f:
        for r, score, output in zip(records, scores, outputs):
            result = {
                "prompt_id": r["prompt_id"],
                "prompt": r["prompt"],
                "response": r["response"],
                "score": score,
                "llama_guard_output": output
            }
            f.write(json.dumps(result) + "\n")
    print(f"Wrote {len(records)} responses to {args.output}")


if __name__ == "__main__":
    main()