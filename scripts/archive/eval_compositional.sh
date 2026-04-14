source "/home/localadmin/bz/CLIP-R/model/models.sh"
if [ "${#models[@]}" -ne "${#processors[@]}" ]; then
  echo "models/processors length mismatch: ${#models[@]} vs ${#processors[@]}"
  exit 1
fi


# 循环运行评估
for i in "${!models[@]}"; do
  echo "Evaluating model: ${models[$i]}"
  
  python eval/eval_compostional.py \
    --model_path "${models[$i]}" \
    --processor_path "${processors[$i]}" \
    --device cuda:0 \
    --skip_if_exists \
    --results_dir "eval/results/compositional_results"

  # 等待当前任务完成（如果你想并行跑多个 GPU，可以参考原脚本的 jobs 处理逻辑）
done

for i in "${!models[@]}"; do
  python eval/eval_sugarcrepe_pp.py \
    --model_path "${models[$i]}" \
    --processor_name "${processors[$i]}" \
    --model_name auto \
    --dataset_name Aman-J/SugarCrepe_pp \
    --image_dir /leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/val2017 \
    --batch_size 512 \
    --skip_if_exists \
    --device cuda:0 \
    --results_dir "eval/results/sugarcrepe_pp" &

  while [ "$(jobs -rp | wc -l)" -ge 1 ]; do
    wait -n
  done
done

echo "All evaluations completed."