#!/bin/bash
# CodeV-R1 Agentic RL training with Qwen3.5-9B on AgentGym backend.
#
# Requires verl main (>=0.8.0.dev) + vllm 0.18.0 + transformers 5.3.0
# to support the Qwen3.5 (model_type=qwen3_5, Qwen3_5ForConditionalGeneration)
# architecture.
#
# Based on verl's official example:
#   examples/grpo_trainer/run_qwen3_5_27b_vllm_fsdp.sh
#
# Key differences from run_agentgym_codev_b0.sh (Qwen3-8B version):
#   - model: Qwen3.5-9B (VL, dense, GDN linear attention)
#   - verl main branch (0.8.0.dev) instead of PyPI 0.7.1
#   - vllm 0.18.0, transformers >=5.3.0
#   - FSDP2 strategy with offload configs (param/optimizer/policy)
#   - Reduced memory pressure (ppo_micro=2, gpu_util=0.35)
#
# Usage (Launchpad, single node):
#   lputil submit -z b0 -n 1 --name rllm-codev-qwen35 -- \
#       bash scripts/agent/swe/run_agentgym_codev_qwen35_b0.sh
#
set -ex

# ── 0. Install dependencies (idempotent) ──────────────────────────────
# msh-agentgym: LofiSandbox for SWE sandbox execution
# msh-swebench: swebench harness for reward computation
pip install --quiet \
    "msh-agentgym>=1.4.14" \
    "msh-swebench>=0.2.13" \
    2>/dev/null || true

# Remove the wrong PyPI "agentgym" if it was pre-installed
pip uninstall -y agentgym 2>/dev/null || true

# rllm source is synced to moonfs (accessible from compute nodes).
# Install without [verl] extra first to avoid pulling the old verl 0.7.1 pin.
RLLM_SRC=${RLLM_SRC:-"/mnt/moonfs/chenzhirong-b0/rllm-swe/rllm"}
pip install -e "${RLLM_SRC}"

# Install verl main branch + compatible vllm / transformers for Qwen3.5.
# rllm's [verl] extra pins vllm<=0.12 and verl==0.7.1, both too old for qwen3_5.
# We install the pinned versions here AFTER rllm's base install so these
# overrides win.
# Pin verl to commit b4c82633 (2026-04-03): includes Qwen3.5 FSDP support
# (merged 2026-03-30 in PR #5682) but BEFORE the 2026-04-20 BREAKING refactor
# (PR #6067) that moved fsdp_workers.py → engine_workers.py, which breaks
# rllm's import `from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker`.
pip install --upgrade --no-deps \
    "git+https://github.com/verl-project/verl.git@b4c82633"
# Install vllm 0.17 WITH all its deps (pip will auto-downgrade transformers
# to <5 since vllm's metadata says so). This gives us compressed-tensors,
# xgrammar, flashinfer-python, and all the other vllm runtime deps.
pip install --upgrade "vllm==0.17.0"

# Now force transformers to 5.3 for qwen3_5 architecture support. vllm's
# <5 pin is overly conservative — the verl team's official vllm017.latest
# image ships with transformers 5.3 and it works at runtime.
pip install --upgrade --no-deps --force-reinstall "transformers==5.3.0"

# verl main's own runtime deps (we used --no-deps for the verl install above).
pip install --upgrade accelerate peft pylatexenc torchdata

# Qwen3.5 GDN linear attention + tensordict at version verl expects.
# Install BEFORE the engine_workers sed-patch so `import verl.workers.*` works
# (verl's __init__ imports protocol which imports tensordict).
pip install --upgrade \
    flash-linear-attention \
    "tensordict>=0.8.0,<=0.10.0,!=0.9.0"

# Patch verl's tensordict_utils for DataProto compatibility:
#   1. assign_non_tensor — route DataProto to meta_info (called from
#      engine_workers + engine/fsdp/transformer_impl).
#   2. chunk_tensordict — asserts isinstance(td, TensorDict), fails on DataProto.
#      DataProto has its own .chunk(chunks) method; delegate to that.
VERL_TU=$(python3 -c "import verl.utils.tensordict_utils as m; print(m.__file__)")
echo "Patching ${VERL_TU}"
python3 - <<PYEOF
import pathlib
p = pathlib.Path("${VERL_TU}")
src = p.read_text()

marker = "# _RLLM_PATCHED_ASSIGN_NON_TENSOR_ACCEPTS_DATAPROTO"
if marker not in src:
    old = '    assert isinstance(tensor_dict, TensorDict), "input dict must be a TensorDict"\n    for key, val in kwargs.items():'
    new = (
        f"    {marker}\n"
        "    if hasattr(tensor_dict, 'meta_info') and hasattr(tensor_dict, 'non_tensor_batch'):\n"
        "        tensor_dict.meta_info.update(kwargs)\n"
        "        return tensor_dict\n"
        '    assert isinstance(tensor_dict, TensorDict), "input dict must be a TensorDict"\n'
        "    for key, val in kwargs.items():"
    )
    assert old in src, f"expected assign_non_tensor pattern not found in {p}"
    src = src.replace(old, new, 1)
    print("Patched assign_non_tensor")

marker2 = "# _RLLM_PATCHED_CHUNK_TENSORDICT_ACCEPTS_DATAPROTO"
if marker2 not in src:
    old = '    assert isinstance(td, TensorDict) and len(td) % chunks == 0, (\n        f"expecting td with length divisible by chunks, but got {len(td)} and {chunks}"\n    )'
    new = (
        f"    {marker2}\n"
        "    if hasattr(td, 'meta_info') and hasattr(td, 'non_tensor_batch') and hasattr(td, 'chunk'):\n"
        "        return td.chunk(chunks)\n"
        '    assert isinstance(td, TensorDict) and len(td) % chunks == 0, (\n'
        '        f"expecting td with length divisible by chunks, but got {len(td)} and {chunks}"\n'
        "    )"
    )
    assert old in src, f"expected chunk_tensordict assert pattern not found in {p}"
    src = src.replace(old, new, 1)
    print("Patched chunk_tensordict")

p.write_text(src)
print(f"Wrote {p}")
PYEOF

# Add .keys() method to verl's DataProto class. Multiple places in verl
# treat DataProto as TensorDict and call data.keys() (maybe_fix_3d_position_ids,
# engine_workers.infer_batch, etc.). Add a keys() method that returns all keys
# across the three underlying collections.
VERL_PROTOCOL=$(python3 -c "import verl.protocol as m; print(m.__file__)")
echo "Patching DataProto in ${VERL_PROTOCOL} to add keys() method"
python3 - <<PYEOF
import pathlib
p = pathlib.Path("${VERL_PROTOCOL}")
src = p.read_text()
marker = "# _RLLM_PATCHED_DATAPROTO_KEYS"
if marker not in src:
    # Inject a keys() method into DataProto class. Find the class definition
    # and add the method right after the 'class DataProto' line.
    old = "class DataProto:"
    new = (
        "class DataProto:\n"
        f"    {marker}\n"
        "    def keys(self):\n"
        "        return list(self.batch.keys()) + list(self.non_tensor_batch.keys()) + list(self.meta_info.keys())\n"
        "    def __contains__(self, key):\n"
        "        return key in self.batch.keys() or key in self.non_tensor_batch or key in self.meta_info\n"
        "    def get(self, key, default=None):\n"
        "        if key in self.batch.keys():\n"
        "            return self.batch[key]\n"
        "        if key in self.non_tensor_batch:\n"
        "            return self.non_tensor_batch[key]\n"
        "        if key in self.meta_info:\n"
        "            return self.meta_info[key]\n"
        "        return default\n"
    )
    assert old in src, f"expected 'class DataProto:' not found in {p}"
    src = src.replace(old, new, 1)

# Extend __getitem__ to support string keys (returns from batch/non_tensor/meta).
marker_gi = "# _RLLM_PATCHED_DATAPROTO_GETITEM"
if marker_gi not in src:
    old = '        else:\n            raise TypeError(f"Indexing with {type(item)} is not supported")'
    new = (
        "        elif isinstance(item, str):\n"
        f"            {marker_gi}\n"
        "            if item in self.batch.keys():\n"
        "                return self.batch[item]\n"
        "            if item in self.non_tensor_batch:\n"
        "                return self.non_tensor_batch[item]\n"
        "            if item in self.meta_info:\n"
        "                return self.meta_info[item]\n"
        "            raise KeyError(item)\n"
        "        else:\n"
        '            raise TypeError(f"Indexing with {type(item)} is not supported")'
    )
    assert old in src, f"expected __getitem__ raise pattern not found in {p}"
    src = src.replace(old, new, 1)

p.write_text(src)
print(f"Patched DataProto in {p}")
PYEOF

# Patch verl's engine_workers.infer_batch to skip the buggy tu.pop() call.
# verl main (b4c82633) has an API mismatch: tu.pop expects tensordict-style
# pop(key, default) but DataProto.pop uses list-based pop(batch_keys=[...]).
# Root cause: tu.pop(data, key='no_lora_adapter', default=False) on a
# DataProto iterates chars of the key string and asserts each in batch.keys().
# We're not using LoRA, so hard-code no_lora_adapter=False at the verl source
# level. Source patch applies to all Ray workers (same .py file).
VERL_ENGINE_WORKERS=$(python3 -c "import verl.workers.engine_workers as m; print(m.__file__)")
echo "Patching engine_workers.infer_batch in ${VERL_ENGINE_WORKERS}"
python3 - <<PYEOF
import pathlib
p = pathlib.Path("${VERL_ENGINE_WORKERS}")
src = p.read_text()

# Patch 1: replace buggy tu.pop (DataProto API mismatch, see above).
marker1 = "# _RLLM_PATCHED_NO_LORA_ADAPTER"
if marker1 not in src:
    old = 'no_lora_adapter = tu.pop(data, key="no_lora_adapter", default=False)'
    new = f'no_lora_adapter = False  {marker1}'
    assert old in src, f"expected pattern 1 not found in {p}"
    src = src.replace(old, new)

# Patch 2: replace data.keys() — DataProto has no .keys() method. Check all
# three underlying collections (batch tensordict, non_tensor_batch, meta_info).
marker2 = "# _RLLM_PATCHED_DATA_KEYS"
if marker2 not in src:
    old = "if key not in data.keys():"
    new = (
        "if key not in data.batch.keys() and "
        "key not in data.non_tensor_batch and "
        f"key not in data.meta_info:  {marker2}"
    )
    assert old in src, f"expected pattern 2 not found in {p}"
    src = src.replace(old, new)

# Patch 3: tu.assign_non_tensor asserts TensorDict, fails on DataProto.
# Write directly to meta_info dict instead (it's a plain dict on DataProto).
marker3 = "# _RLLM_PATCHED_ASSIGN_NON_TENSOR"
if marker3 not in src:
    old = "tu.assign_non_tensor(data, **{key: val})"
    new = f"data.meta_info[key] = val  {marker3}"
    assert old in src, f"expected pattern 3 not found in {p}"
    src = src.replace(old, new)

p.write_text(src)
print(f"Patched {p}")
PYEOF

# flash-attn ABI is a moving target against the Moonshot-custom torch build.
# Every version we tried (base image's 2.8.1+msh, pypi.msh.team's
# 2.8.1+msh.9230329.torch29cu129.cxx11.abi) fails to resolve c10::cuda
# symbols at import time. Bypass the problem entirely: patch the model's
# config.json to force _attn_implementation=sdpa before training starts.
# transformers then skips the flash_attn check and uses torch's built-in
# scaled_dot_product_attention for the full_attention layers. The GDN
# (linear_attention) layers use flash-linear-attention independently.
python3 -c "
import json, sys
cfg_path = '${MODEL}/config.json' if '${MODEL}' else '/mnt/moonfs/chenzhirong-b0/model/Qwen3.5-9B/config.json'
with open(cfg_path) as f:
    cfg = json.load(f)
cfg['_attn_implementation'] = 'sdpa'
if 'text_config' in cfg:
    cfg['text_config']['_attn_implementation'] = 'sdpa'
with open(cfg_path, 'w') as f:
    json.dump(cfg, f, indent=2)
print(f'Forced _attn_implementation=sdpa in {cfg_path}')
"

# flash-attn wheel from PyPI was built against stock torch ABI, but the base
# image ships a Moonshot-custom torch build (torch 2.10+cu129.msh). The ABI
# mismatch gives:
#   ImportError: flash_attn_2_cuda...undefined symbol: _ZN3c104cuda29c10_cuda_check_implementation...
# Uninstall the broken flash-attn so transformers falls back to torch's
# built-in SDPA (slower than flash-attn-2 but has no ABI dependency).
# flash-linear-attention is Triton-based and unaffected.
pip uninstall -y flash-attn 2>/dev/null || true

# The image's cuDNN install is broken (CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED
# on conv3d / any cuDNN backend call). verl's qwen3_5 wrapper does a
# dummy visual tower forward for text-only batches (FSDP param-touch trick),
# which hits F.conv3d → cuDNN → crash. Disable cuDNN globally via
# sitecustomize.py so every Python process in this job (driver + Ray workers)
# picks it up. Torch falls back to native CUDA kernels (slower conv, same
# matmul since matmul uses cuBLAS not cuDNN).
SITECUSTOMIZE=$(python3 -c "import site; print(site.getsitepackages()[0])")/sitecustomize.py
cat > "${SITECUSTOMIZE}" <<'SITECUSTOMIZE_EOF'
# _RLLM_DISABLE_CUDNN — injected by run_agentgym_codev_qwen35_b0.sh
try:
    import torch
    torch.backends.cudnn.enabled = False
except Exception:
    pass
SITECUSTOMIZE_EOF
echo "Wrote ${SITECUSTOMIZE} (disables cuDNN globally)"

# ── 1. Environment variables ──────────────────────────────────────────
: "${WANDB_API_KEY:?WANDB_API_KEY must be set}"
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000
# Extend NCCL collective timeout to 30 min to tolerate FSDP all_gather stragglers.
export NCCL_TIMEOUT=1800000
export TORCH_NCCL_BLOCKING_WAIT=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800

RLLM_DIR=$(python3 -c "import rllm; import os; print(os.path.dirname(os.path.dirname(rllm.__file__)))")

# ── 2. Data paths ─────────────────────────────────────────────────────
TRAIN_DATA=${TRAIN_DATA:-"/mnt/moonfs/chenzhirong-b0/rllm-swe/data/codev_r1_train1519.parquet"}
VAL_DATA=${VAL_DATA:-"/mnt/moonfs/chenzhirong-b0/rllm-swe/data/codev_r1_val32.parquet"}
CKPT_DIR=${CKPT_DIR:-"/mnt/moonfs/chenzhirong-b0/rllm-swe/checkpoints/test_codev_r1_qwen35_9b"}
ROLLOUT_DIR=${ROLLOUT_DIR:-"/mnt/moonfs/chenzhirong-b0/rllm-swe/rollouts/test_codev_r1_qwen35_9b"}

# ── 2.5 Multi-node Ray cluster setup ───────────────────────────────────
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
RAY_PORT=${RAY_PORT:-6379}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}

if [ "${NNODES}" -gt 1 ]; then
    if [ "${NODE_RANK}" == "0" ]; then
        echo "[ray] Starting Ray head on node-0 (${MASTER_ADDR}:${RAY_PORT})"
        ray start --head --port=${RAY_PORT} --num-gpus=${GPUS_PER_NODE} --disable-usage-stats
        echo "[ray] Waiting for ${NNODES} nodes to join..."
        until [ "$(ray status 2>/dev/null | grep -c '1.0/8.0 GPU')" -ge ${NNODES} ] || \
              [ "$(ray status 2>/dev/null | grep -E '^\s+[0-9]+ node' | awk '{print $1}')" -ge ${NNODES} ]; do
            n=$(ray status 2>/dev/null | grep -E "node_" | wc -l)
            if [ "$n" -ge "${NNODES}" ]; then break; fi
            echo "[ray] Currently ${n}/${NNODES} nodes ready, waiting..."
            sleep 5
        done
        echo "[ray] All ${NNODES} nodes joined. Starting training."
    else
        echo "[ray] Starting Ray worker on node-${NODE_RANK}"
        sleep 15
        ray start --address=${MASTER_ADDR}:${RAY_PORT} --num-gpus=${GPUS_PER_NODE} --disable-usage-stats --block
        exit 0
    fi
fi

# ── 3. Launch training (only on node-0 for multi-node) ────────────────
#
# Memory budget (H200 / 141GB per GPU, Qwen3.5-9B + vision tower ~2B extra):
#   vLLM KV cache (0.35 util)           : ~49 GB
#   FSDP2 model shard (offloaded)       : ~0 GB during rollout (CPU offload)
#   FSDP2 model shard (resident, train) : ~10 GB
#   Optimizer shard (offloaded)         : ~0 GB (CPU offload)
#   Vision tower weights (resident)     : ~4 GB
#   Gradients (sharded)                 : ~1 GB
#   Activations (micro=2, seq=12K)      : ~15-20 GB
#   ─────────────────────────────────────────────
#   Peak ≈ 80-90 GB (57-64% utilization)
python3 -m rllm.trainer.verl.train_agent_ppo \
    algorithm.adv_estimator=rloo \
    data.train_files="${TRAIN_DATA}" \
    data.val_files="${VAL_DATA}" \
    data.train_batch_size=8 \
    data.val_batch_size=16 \
    data.max_prompt_length=4096 \
    data.max_response_length=12288 \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=32 \
    actor_rollout_ref.model.path=${MODEL:-"/mnt/moonfs/chenzhirong-b0/model/Qwen3.5-9B"} \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=8 \
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True \
    actor_rollout_ref.actor.fsdp_config.offload_policy=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.fsdp_config.reshard_after_forward=True \
    actor_rollout_ref.ref.fsdp_config.offload_policy=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.use_torch_compile=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode="async" \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    +actor_rollout_ref.rollout.enable_sleep_mode=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0 \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    rllm.mask_truncated_samples=False \
    trainer.critic_warmup=0 \
    trainer.use_legacy_worker_impl=enable \
    trainer.logger=['console','wandb'] \
    trainer.project_name='rllm-codev-agentgym' \
    trainer.experiment_name='test_codev_r1_qwen35_9b' \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=${NNODES:-1} \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir="${CKPT_DIR}" \
    trainer.log_val_generations=20 \
    trainer.rollout_data_dir="${ROLLOUT_DIR}" \
    rllm.env.name=agentgym_swe \
    rllm.agent.name=sweagent \
    +rllm.agent.agent_args.scaffold=coding \
    rllm.agent.max_steps=20 \
    rllm.agent.overlong_filter=True \
    rllm.agent.trajectory_timeout=5400 \
    trainer.total_epochs=5
