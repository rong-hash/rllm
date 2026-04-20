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
pip install --upgrade --no-deps \
    "git+https://github.com/verl-project/verl.git@main"
pip install --upgrade \
    "vllm==0.18.0" \
    "transformers>=5.3.0" \
    "flash-linear-attention" \
    "tensordict>=0.8.0,<=0.10.0,!=0.9.0"

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
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
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
    trainer.logger=['console','wandb'] \
    trainer.project_name='rllm-codev-agentgym' \
    trainer.experiment_name='test_codev_r1_qwen35_9b' \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=${NNODES:-1} \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=/mnt/moonfs/chenzhirong-b0/rllm-swe/checkpoints/test_codev_r1_qwen35_9b \
    trainer.log_val_generations=20 \
    trainer.rollout_data_dir=/mnt/moonfs/chenzhirong-b0/rllm-swe/rollouts/test_codev_r1_qwen35_9b \
    rllm.env.name=agentgym_swe \
    rllm.agent.name=sweagent \
    +rllm.agent.agent_args.scaffold=coding \
    rllm.agent.max_steps=20 \
    rllm.agent.overlong_filter=True \
    rllm.agent.trajectory_timeout=5400 \
    trainer.total_epochs=5
