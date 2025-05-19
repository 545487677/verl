set -x

export VLLM_ATTENTION_BACKEND=XFORMERS
ray stop                         
sleep 1s 
# export RAY_DEBUG_POST_MORTEM=1
export HYDRA_FULL_ERROR=1
pip install --force-reinstall psutil==5.9.8
pip install -U "ray[data,train,tune,serve]"
# conda install -c conda-forge rdkit -y
# If you are using vllm<=0.6.3, you might need to set the following environment variable to avoid bugs:
# export VLLM_ATTENTION_BACKEND=XFORMERS



python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/vepfs/fs_projects/FunMG/LLM/dataset/mol_grpo/desc2mol_grpo_parquet/train.parquet \
    data.val_files=/vepfs/fs_projects/FunMG/LLM/dataset/mol_grpo/desc2mol_grpo_parquet/test.parquet \
    data.train_batch_size=64 \
    data.max_prompt_length=512 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=/vepfs/fs_projects/FunMG/LLM/model_weight/qwen/Qwen2.5-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=3e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=160 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=160 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_grpo_hme_desc2mol_format_acc_t07' \
    trainer.experiment_name='qwen2_7b_function_rm' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=5 \
    trainer.total_epochs=5  2>&1 | tee verl_hme_qwen25_7b_t07_demo.log

    ## 64g * 2

cd /vepfs/fs_ckps/guojianz/llm/verl_exp/verl_grpo_hme_desc2mol_format_acc_t07/qwen2_7b_function_rm
max_step=$(ls -d global_step_* | sed 's/global_step_//' | sort -n | tail -1)
max_dir="global_step_$max_step"

for d in global_step_*; do
  if [ "$d" != "$max_dir" ] && [ -d "$d" ]; then
    echo "Deleting $d"
    rm -rf "$d"
  fi
done