"""Convert mshrl SWE data (jsonl) to rllm parquet format.

mshrl format (jsonl, one JSON object per line)::

    {
        "dataset": "swebench-verified",
        "instance_id": "django__django-16493",
        "prompt": "We are currently solving ...",
        "stage": "repair",
        "gt_rollout": { ... },
        ...  // any extra fields are preserved
    }

rllm format (parquet)::

    data_source | prompt (list[dict]) | ability | reward_model | extra_info (json str)

Usage::

    python scripts/data/convert_mshrl_swe.py \\
        --input /path/to/swebench_train.jsonl \\
        --output /path/to/output.parquet \\
        [--scaffold r2egym]

Multiple input files can be specified (they will be concatenated).
"""

import argparse
import json
import os
import sys

import pandas as pd

# Append rllm root so we can import system prompts.
RLLM_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, RLLM_DIR)

from rllm.agents.system_prompts import (
    SWE_SYSTEM_PROMPT,
    SWE_USER_PROMPT,
    SWEAGENT_SYSTEM_PROMPT,
    SWEAGENT_USER_PROMPT,
)

SCAFFOLD_PROMPTS = {
    "r2egym": (SWE_SYSTEM_PROMPT, SWE_USER_PROMPT),
    "sweagent": (SWEAGENT_SYSTEM_PROMPT, SWEAGENT_USER_PROMPT),
}


def convert_entry(row: dict, system_prompt: str, user_prompt_template: str) -> dict:
    """Convert a single mshrl entry to rllm format."""
    # Extract problem statement — field name varies across data sources.
    problem_statement = row.get("problem_statement", "") or row.get("prompt", "")

    # Build rllm-style chat messages.
    user_content = user_prompt_template.format(problem_statement=problem_statement)
    prompt_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    # Build extra_info — preserve all original fields and ensure key fields exist.
    extra_info = dict(row)
    extra_info["problem_statement"] = problem_statement

    return {
        "data_source": "swe",
        "prompt": prompt_messages,
        "ability": "swe",
        "reward_model": {"style": "rule", "ground_truth": ""},
        # verl 0.7.1 expects extra_info as dict (not json string)
        "extra_info": extra_info,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert mshrl SWE jsonl to rllm parquet."
    )
    parser.add_argument(
        "--input", "-i",
        nargs="+",
        required=True,
        help="Input jsonl file(s) in mshrl format.",
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output parquet path.",
    )
    parser.add_argument(
        "--scaffold",
        choices=["r2egym", "sweagent"],
        default="r2egym",
        help="Scaffold type — determines system/user prompt (default: r2egym).",
    )
    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=0,
        help="Max number of entries to convert (0 = all).",
    )
    args = parser.parse_args()

    system_prompt, user_prompt_template = SCAFFOLD_PROMPTS[args.scaffold]

    rows = []
    for path in args.input:
        with open(path, "r") as f:
            for line in f:
                if args.limit and len(rows) >= args.limit:
                    break
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                rows.append(convert_entry(entry, system_prompt, user_prompt_template))
        print(f"Read {len(rows)} entries from {path}")
        if args.limit and len(rows) >= args.limit:
            break

    if not rows:
        print("No data to convert.")
        return

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    df.to_parquet(args.output)
    print(f"Saved {len(df)} entries to {args.output}")

    # Summary.
    extra_infos = [r["extra_info"] for r in rows]
    datasets = set(e.get("dataset", "?") for e in extra_infos)
    print(f"Datasets: {datasets}")
    print(f"Scaffold: {args.scaffold}")


if __name__ == "__main__":
    main()
