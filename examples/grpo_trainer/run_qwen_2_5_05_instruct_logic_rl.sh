set -x


export VLLM_ATTENTION_BACKEND=XFORMERS
ray stop                         
sleep 1s 
export RAY_DEBUG_POST_MORTEM=1
export HYDRA_FULL_ERROR=1
pip install --force-reinstall psutil==5.9.8
pip install -U "ray[data,train,tune,default]"
pip install debugpy==1.8.0
# conda install -c conda-forge rdkit -y


# start head node
# export RAY_DEBUG=legacy
# ray start --head --node-ip-address 0.0.0.0 --num-gpus 2 --ray-debugger-external --port 6380
# start worker node
# ray start --address='0.0.0.0:6380' --ray-debugger-external

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/vepfs/fs_users/guojianz/dp_project/LLM/LLM_Finetune/Finetune_learning/Logic-RL/data/kk/instruct/3ppl/train.parquet \
    data.val_files=/vepfs/fs_users/guojianz/dp_project/LLM/LLM_Finetune/Finetune_learning/Logic-RL/data/kk/instruct/3ppl/test.parquet \
    data.train_batch_size=4 \
    data.val_batch_size=4 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=/vepfs/fs_projects/FunMG/LLM/model_weight/models--Qwen--Qwen2.5-0.5B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=40 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.val_before_train=True \
    trainer.logger=['console','wandb'] \
    trainer.project_name='GRPO_logic_KK' \
    trainer.experiment_name='Qwen-7B' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=1  2>&1 | tee verl_demo.log

cd /vepfs/fs_ckps/guojianz/llm/verl_exp/GRPO_logic_KK/Qwen-7B
max_step=$(ls -d global_step_* | sed 's/global_step_//' | sort -n | tail -1)
max_dir="global_step_$max_step"

for d in global_step_*; do
  if [ "$d" != "$max_dir" ] && [ -d "$d" ]; then
    echo "Deleting $d"
    rm -rf "$d"
  fi
done