import argparse
import json
import os

from tqdm import tqdm

from utils.runtime import chunked, load_model, load_prompts, set_seed


def get_args():
    parser = argparse.ArgumentParser(
        description="No-steering baseline generation. Loads the model and runs the same "
                    "prompts as CLE-P / CLE-A but registers no hooks."
    )
    parser.add_argument("--model_name", type=str, default="llama3-8b")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dataset", type=str, default="harmbench_test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = get_args()
    set_seed(args.seed)

    model = load_model(args)
    prompts, categories = load_prompts(args)

    out_dir = args.out_dir or os.path.join("./completions", args.model_name, "baseline")
    os.makedirs(out_dir, exist_ok=True)
    limit_str = f"limit{args.limit}" if args.limit else "FULL"
    out_path = os.path.join(
        out_dir,
        f"completions_{args.dataset}_{limit_str}_baseline_seed{args.seed}.json",
    )

    print("--- Baseline Configuration ---")
    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.dataset} | Prompts: {len(prompts)}")
    print(f"Output: {out_path}")

    results = []
    pbar = tqdm(total=len(prompts), desc="Baseline generate")
    for batch_start, batch_prompts in chunked(prompts, args.batch_size):
        batch_categories = categories[batch_start:batch_start + len(batch_prompts)]
        try:
            responses = model.batch_generate(batch_prompts, max_new_tokens=args.max_new_tokens)
        except Exception as e:
            print(f"Batch generation error. Falling back to single-prompt generation: {e}")
            responses = []
            for prompt in batch_prompts:
                try:
                    responses.append(model.generate(prompt, max_new_tokens=args.max_new_tokens))
                except Exception as inner_e:
                    print(f"Gen Error: {inner_e}")
                    responses.append("")
        for prompt, response, category in zip(batch_prompts, responses, batch_categories):
            results.append({"category": category, "prompt": prompt, "response": response})
        pbar.update(len(batch_prompts))
    pbar.close()

    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nSaved baseline completions to {out_path}")


if __name__ == "__main__":
    main()
