# Tested with 2 & 4 GPUs

set -x

# if [ "$#" -lt 2 ]; then
#     echo "Usage: run_gemma_2b.sh <nproc_per_node>  [other_configs...]"
#     exit 1
# fi

nproc_per_node=2
save_path=/vepfs/fs_ckps/guojianz/llm/verl_exp

# Shift the arguments so $@ refers to the rest
shift 2

# /vepfs/fs_projects/FunMG/LLM/model_weight/models--Qwen--Qwen2.5-0.5B-Instruct
# /vepfs/fs_projects/FunMG/LLM/model_weight/qwen/Qwen2___5-7B-Instruct

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=/vepfs/fs_projects/FunMG/LLM/dataset/gsm8k/train.parquet \
    data.val_files=/vepfs/fs_projects/FunMG/LLM/dataset/gsm8k/test.parquet \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    data.prompt_dict_keys=['question'] \
    +data.response_dict_keys=['answer'] \
    data.train_batch_size=32 \
    data.micro_batch_size_per_gpu=4 \
    model.partial_pretrain=/vepfs/fs_projects/FunMG/LLM/model_weight/qwen/Qwen2___5-7B-Instruct \
    trainer.default_local_dir=$save_path \
    trainer.project_name=gsm8k-sft \
    trainer.experiment_name=gsm8k-sft-qwen2_5_7b-it \
    trainer.total_epochs=2 \
    trainer.logger=['console','wandb'] \
    trainer.default_hdfs_dir=null $@