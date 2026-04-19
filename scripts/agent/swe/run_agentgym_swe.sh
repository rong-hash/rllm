#!/bin/bash
# SWE Agentic RL training with AgentGym backend.
#
# Usage (local test, single node):
#   bash scripts/agent/swe/run_agentgym_swe.sh
#
# Usage (Launchpad):
#   lputil submit -z b0h200 -n 8 --name rllm-swe-agentgym -- \
#       bash scripts/agent/swe/run_agentgym_swe.sh
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

# rllm source is synced to moonfs (accessible from compute nodes)
# Install with [verl] extra to pull in verl + vllm
RLLM_SRC=${RLLM_SRC:-"/mnt/moonfs/chenzhirong-ksyun/rllm-swe/rllm"}
pip install -e "${RLLM_SRC}[verl]"

# ── 1. Environment variables ──────────────────────────────────────────
: "${WANDB_API_KEY:?WANDB_API_KEY must be set}"
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000
# Extend NCCL collective timeout to 30 min (default 10 min) to tolerate
# FSDP all_gather stragglers when sequence lengths are imbalanced.
export NCCL_TIMEOUT=1800000
export TORCH_NCCL_BLOCKING_WAIT=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800

RLLM_DIR=$(python3 -c "import rllm; import os; print(os.path.dirname(os.path.dirname(rllm.__file__)))")

# ── 2. Data paths ─────────────────────────────────────────────────────
TRAIN_DATA=${TRAIN_DATA:-"/mnt/moonfs/chenzhirong-ksyun/rllm-swe/data/swe_factory_1000.parquet"}
VAL_DATA=${VAL_DATA:-"/mnt/moonfs/chenzhirong-ksyun/rllm-swe/data/swe_factory_1000.parquet"}

# ── 2.5 Multi-node Ray cluster setup ───────────────────────────────────
# Launchpad injects NNODES, NODE_RANK, MASTER_ADDR for all nodes.
# For N>1 nodes, we need one Ray head + (N-1) workers before launching
# training on node-0 only. Single-node just calls ray.init() normally.
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
RAY_PORT=${RAY_PORT:-6379}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}

if [ "${NNODES}" -gt 1 ]; then
    if [ "${NODE_RANK}" == "0" ]; then
        echo "[ray] Starting Ray head on node-0 (${MASTER_ADDR}:${RAY_PORT})"
        ray start --head --port=${RAY_PORT} --num-gpus=${GPUS_PER_NODE} --disable-usage-stats
        # Wait until all workers have joined
        echo "[ray] Waiting for ${NNODES} nodes to join..."
        until [ "$(ray status 2>/dev/null | grep -c '1.0/8.0 GPU')" -ge ${NNODES} ] || \
              [ "$(ray status 2>/dev/null | grep -E '^\s+[0-9]+ node' | awk '{print $1}')" -ge ${NNODES} ]; do
            # Simpler: just count nodes
            n=$(ray status 2>/dev/null | grep -E "node_" | wc -l)
            if [ "$n" -ge "${NNODES}" ]; then break; fi
            echo "[ray] Currently ${n}/${NNODES} nodes ready, waiting..."
            sleep 5
        done
        echo "[ray] All ${NNODES} nodes joined. Starting training."
    else
        echo "[ray] Starting Ray worker on node-${NODE_RANK}, connecting to ${MASTER_ADDR}:${RAY_PORT}"
        # Wait for head to be up
        sleep 15
        ray start --address=${MASTER_ADDR}:${RAY_PORT} --num-gpus=${GPUS_PER_NODE} --disable-usage-stats --block
        # --block makes this process stay alive; it'll be killed when job ends
        exit 0
    fi
fi

# ── 3. Launch training (only on node-0 for multi-node) ────────────────
python3 -m rllm.trainer.verl.train_agent_ppo \
    algorithm.adv_estimator=rloo \
    data.train_files="${TRAIN_DATA}" \
    data.val_files="${VAL_DATA}" \
    data.train_batch_size=8 \
    data.val_batch_size=16 \
    data.max_prompt_length=4096 \
    data.max_response_length=16384 \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=32 \
    actor_rollout_ref.model.path=${MODEL:-"/mnt/moonfs/public-models-ksyun/Qwen/Qwen3-8B"} \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode="async" \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.enable_sleep_mode=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    rllm.mask_truncated_samples=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='rllm-swe-agentgym' \
    trainer.experiment_name='test_rllm' \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=${NNODES:-1} \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=/mnt/moonfs/chenzhirong-ksyun/rllm-swe/checkpoints/test_rllm \
    rllm.env.name=agentgym_swe \
    rllm.agent.name=sweagent \
    rllm.agent.max_steps=20 \
    rllm.agent.overlong_filter=True \
    rllm.agent.trajectory_timeout=5400 \
    trainer.total_epochs=1000
