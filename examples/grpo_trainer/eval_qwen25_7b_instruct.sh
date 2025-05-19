MODEL_PATH=/vepfs/fs_ckps/guojianz/llm/verl_exp/verl_grpo_example_gsm8k_reimplement/qwen2_7b_function_rm/merged_hf_model
DATA_PATH=/vepfs/fs_projects/FunMG/LLM/dataset/gsm8k


# Generation
python3 -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=1 \
    data.path=$DATA_PATH/test.parquet \
    data.prompt_key=prompt \
    data.batch_size=512 \
    data.n_samples=1 \
    data.output_path=$DATA_PATH/test-output-8.parquet \
    model.path=$MODEL_PATH \
    rollout.temperature=0.6 \
    rollout.top_p=0.95 \
    rollout.prompt_length=512 \
    rollout.response_length=1024 \
    rollout.tensor_model_parallel_size=1 \
    rollout.gpu_memory_utilization=0.9 \
    rollout.max_num_batched_tokens=65536

# # Evaluation
# python3 -m recipe.r1.main_eval \
#     data.path=$DATA_PATH/test-output-8.parquet \
#     data.prompt_key=prompt \
#     data.response_key=responses \
#     custom_reward_function.path=recipe/r1/reward_score.py \
#     custom_reward_function.name=reward_func
