export OMP_NUM_THREADS=8
export PYTHONPATH="/home/localadmin/bz/CLIP-R/llava_next:${PYTHONPATH}"


LLM_VERSION="Qwen/Qwen3-1.7B"
LLM_VERSION_CLEAN="Qwen3-1.7B"
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="siglip-so14"

PROMPT_VERSION=plain

IMAGE_FOLDER="/home/localadmin/bz/CLIP-R/data/llava-pretrain-data/images"
DATA_PATH="/home/localadmin/bz/CLIP-R/data/llava-pretrain-data/blip_laion_cc_sbu_558k.json"
BASE_RUN_NAME="llavanext-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-mlp2x_gelu-pretrain_blip558k_plain"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

OUTPUT_DIR="/home/localadmin/bz/CLIP-R/llava_next/checkpoints/projectors/${BASE_RUN_NAME}"
mkdir -p $OUTPUT_DIR

export CUDA_VISIBLE_DEVICES=0,2
accelerate launch \
  --mixed_precision=bf16 \
  --num_machines 1 \
  --num_processes 2 \
  --machine_rank 0 \
  --main_process_ip "localhost" \
  --main_process_port 29501 \
 /home/localadmin/bz/CLIP-R/llava_next/llava/train/train_mem.py \
    --deepspeed /home/localadmin/bz/CLIP-R/llava_next/scripts/zero2.json \
    --model_name_or_path ${LLM_VERSION} \
    --version ${PROMPT_VERSION} \
    --data_path ${DATA_PATH} \
    --image_folder ${IMAGE_FOLDER} \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_tunable_parts="mm_mlp_adapter" \
    --mm_vision_select_layer -2 \
    --mm_projector_type mlp2x_gelu \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --eval_strategy "no" \
    --save_strategy "no" \
    --save_steps 50000 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name $BASE_RUN_NAME \
    --attn_implementation sdpa