"""Convert CodeV-R1 task directories to rllm parquet format.

Input: a directory containing N task subdirs (codev-NNNNN/) with layout::

    codev-NNNNN/
    ├── task.toml                          # metadata (instance_id, task_type)
    ├── instruction.md                     # problem statement
    ├── environment/
    │   ├── Dockerfile                     # identical across all tasks
    │   └── testbed/
    │       ├── rtl/                       # empty (agent writes solution.v here)
    │       └── verif/
    │           ├── gold.v                 # per-task reference implementation
    │           ├── port_info.json         # per-task interface spec
    │           └── run_test.py            # identical across all tasks
    └── tests/test.sh                      # identical across all tasks

Output: parquet compatible with AgentGymSWEEnv. Each row carries everything the
env needs (gold.v / port_info.json / test.sh / run_test.py) so a shared base
image with ``hdlc/sim:osvb`` + python3 + git is sufficient at runtime; the env
must inject these files into the sandbox before evaluation.

Usage::

    python scripts/data/convert_codev_r1.py \\
        --input /mnt/moonfs/chenzhirong-ksyun/codev-r1-task \\
        --output /mnt/moonfs/chenzhirong-ksyun/rllm-swe/data/codev_r1_1551.parquet
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import tomllib

RLLM_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, RLLM_DIR)

from rllm.agents.system_prompts import (
    CODING_SYSTEM_PROMPT,
    CODING_USER_PROMPT,
    SWE_SYSTEM_PROMPT,
    SWE_USER_PROMPT,
    SWEAGENT_SYSTEM_PROMPT,
    SWEAGENT_USER_PROMPT,
)

SCAFFOLD_PROMPTS = {
    "coding": (CODING_SYSTEM_PROMPT, CODING_USER_PROMPT),
    "r2egym": (SWE_SYSTEM_PROMPT, SWE_USER_PROMPT),
    "sweagent": (SWEAGENT_SYSTEM_PROMPT, SWEAGENT_USER_PROMPT),
}

def read_task(task_dir: Path) -> dict | None:
    """Read one codev-NNNNN task directory into an rllm parquet row.

    Returns None if the task directory is malformed.
    """
    task_toml = task_dir / "task.toml"
    instruction_md = task_dir / "instruction.md"
    dockerfile = task_dir / "environment" / "Dockerfile"
    gold_v = task_dir / "environment" / "testbed" / "verif" / "gold.v"
    port_info_json = task_dir / "environment" / "testbed" / "verif" / "port_info.json"
    run_test_py = task_dir / "environment" / "testbed" / "verif" / "run_test.py"
    test_sh = task_dir / "tests" / "test.sh"

    for required in (task_toml, instruction_md, gold_v, port_info_json, run_test_py, test_sh):
        if not required.is_file():
            print(f"[skip] {task_dir.name}: missing {required.name}", file=sys.stderr)
            return None

    with open(task_toml, "rb") as f:
        meta = tomllib.load(f)
    instance_id = meta.get("metadata", {}).get("instance_id", task_dir.name)
    task_type = meta.get("metadata", {}).get("task_type", "verilog_generation")
    verifier_timeout = meta.get("verifier", {}).get("timeout_sec", 120.0)
    agent_timeout = meta.get("agent", {}).get("timeout_sec", 1800.0)

    problem_statement = instruction_md.read_text()

    return {
        "instance_id": instance_id,
        "task_type": task_type,
        "problem_statement": problem_statement,
        "gold_v": gold_v.read_text(),
        "port_info": port_info_json.read_text(),
        "run_test_py": run_test_py.read_text(),
        "eval_sh": test_sh.read_text(),
        "verifier_timeout": verifier_timeout,
        "agent_timeout": agent_timeout,
    }


def build_row(task: dict, scaffold: str) -> dict:
    """Wrap a task dict into the rllm parquet schema."""
    system_prompt, user_prompt_template = SCAFFOLD_PROMPTS[scaffold]

    user_content = user_prompt_template.format(problem_statement=task["problem_statement"])
    prompt_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    extra_info = {
        "dataset": "codev-r1",
        "instance_id": task["instance_id"],
        "task_type": task["task_type"],
        "language": "verilog",
        "problem_statement": task["problem_statement"],
        # Runtime-injected files (env must write these to the sandbox on reset)
        "gold_v": task["gold_v"],
        "port_info": task["port_info"],
        "run_test_py": task["run_test_py"],
        "eval_sh": task["eval_sh"],
        # Fallback image (public DockerHub). Env may ignore and use DOCKER_CONFIG.
        "docker_image": "hdlc/sim:osvb",
        "workspace_dir": "/testbed",
        # Timeouts from task.toml
        "verifier_timeout": task["verifier_timeout"],
        "agent_timeout": task["agent_timeout"],
    }

    return {
        "data_source": "swe",
        "prompt": prompt_messages,
        "ability": "swe",
        "reward_model": {"style": "rule", "ground_truth": ""},
        "extra_info": extra_info,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert CodeV-R1 tasks to rllm parquet.")
    parser.add_argument("--input", "-i", required=True, help="Dir containing codev-NNNNN/ subdirs.")
    parser.add_argument("--output", "-o", required=True, help="Output parquet path.")
    parser.add_argument(
        "--scaffold",
        choices=["coding", "r2egym", "sweagent"],
        default="coding",
        help="Prompt scaffold (default: coding — generic code-agent framing, no github-issue assumptions).",
    )
    parser.add_argument("--limit", "-n", type=int, default=None, help="Max tasks to convert.")
    args = parser.parse_args()

    input_dir = Path(args.input)
    if not input_dir.is_dir():
        print(f"Input dir not found: {input_dir}", file=sys.stderr)
        return 1

    task_dirs = sorted(d for d in input_dir.iterdir() if d.is_dir() and d.name.startswith("codev-"))
    if args.limit is not None:
        task_dirs = task_dirs[: args.limit]

    print(f"Found {len(task_dirs)} candidate task dirs.")

    rows: list[dict] = []
    skipped = 0
    for i, task_dir in enumerate(task_dirs, start=1):
        task = read_task(task_dir)
        if task is None:
            skipped += 1
            continue
        rows.append(build_row(task, args.scaffold))
        if i % 200 == 0:
            print(f"  [{i}/{len(task_dirs)}] {task_dir.name}")

    print(f"Converted {len(rows)} tasks ({skipped} skipped).")

    if not rows:
        print("No rows produced, aborting.", file=sys.stderr)
        return 1

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(output_path, engine="pyarrow", index=False)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Wrote {len(df)} rows → {output_path} ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
