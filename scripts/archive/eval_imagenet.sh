source "/home/localadmin/bz/CLIP-R/model/models.sh"

if [ "${#models[@]}" -ne "${#processors[@]}" ]; then
  echo "models/processors length mismatch: ${#models[@]} vs ${#processors[@]}"
  exit 1
fi

for i in "${!models[@]}"; do
  python eval/eval_zeroshot_imagenet.py \
    --model_path "${models[$i]}" \
    --processor_path "${processors[$i]}" \
    --dataset all \
    --batch_size 256 \
    --num_workers 8 \
    --device cuda:0 \
    --skip_if_exists \
    --results_dir "$WORK/fmohamma/CLIP-R/eval/results/classification_imagenet" &

  while [ "$(jobs -rp | wc -l)" -ge 1 ]; do
    wait -n
  done
done

wait
