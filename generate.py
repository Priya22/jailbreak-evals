"""Run a prompt file through a model with vLLM and save responses keyed by prompt_id.

Input  (--prompts): JSONL, one object per line: {"prompt_id": ..., "prompt": "..."}
                    (a .json file containing a list of such objects also works)
Output (--output):  JSONL: {"prompt_id": ..., "prompt": ..., "response": ...}

Example:
    python generate.py --model meta-llama/Llama-3.1-8B-Instruct \
        --prompts prompts.jsonl --output responses.jsonl
"""
import argparse
import json
from pathlib import Path

from vllm import LLM, SamplingParams


def load_prompts(path):
    text = Path(path).read_text(encoding="utf-8")
    if path.endswith(".json"):
        records = json.loads(text)
    else:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    ids = set()
    for i, r in enumerate(records):
        if "prompt_id" not in r or "prompt" not in r:
            raise ValueError(f"Record {i} must have 'prompt_id' and 'prompt' keys")
        if r["prompt_id"] in ids:
            raise ValueError(f"Duplicate prompt_id: {r['prompt_id']}")
        ids.add(r["prompt_id"])
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="HF model name or local path")
    p.add_argument("--prompts", required=True, help="Prompt file (.jsonl or .json)")
    p.add_argument("--output", required=True, help="Output JSONL file")
    p.add_argument("--no-chat-template", action="store_true",
                   help="Feed prompts as raw text instead of applying the chat template")
    p.add_argument("--system-prompt", default=None, help="Optional system prompt (chat mode)")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--max-model-len", type=int, default=32000)
    p.add_argument("--dtype", default="auto")
    args = p.parse_args()

    records = load_prompts(args.prompts)
    print("Loaded {} prompts".format(len(records)))

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
        seed=args.seed or 0,
    )
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    print("LLM {} loaded".format(args.model))

    if args.no_chat_template:
        outputs = llm.generate([r["prompt"] for r in records], sampling)
    else:
        convos = []
        for r in records:
            msgs = []
            if args.system_prompt:
                msgs.append({"role": "system", "content": args.system_prompt})
            msgs.append({"role": "user", "content": r["prompt"]})
            convos.append(msgs)
        outputs = llm.chat(convos, sampling)

    # vLLM returns outputs in input order; pair them back by index -> prompt_id.
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r, o in zip(records, outputs):
            f.write(json.dumps({
                "prompt_id": r["prompt_id"],
                "prompt": r["prompt"],
                "response": o.outputs[0].text,
            }, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} responses to {out_path}")


if __name__ == "__main__":
    main()
