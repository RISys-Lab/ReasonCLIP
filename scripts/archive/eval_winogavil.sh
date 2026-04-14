source "/home/localadmin/bz/CLIP-R/model/models.sh"
if [ "${#models[@]}" -ne "${#processors[@]}" ]; then
  echo "models/processors length mismatch: ${#models[@]} vs ${#processors[@]}"
  exit 1
fi

for i in "${!models[@]}"; do
  python eval/eval_winogavil.py \
    --model_path "${models[$i]}" \
    --processor_path "${processors[$i]}" \
    --skip_if_exists \
    --batch_size 32 \
    --results_dir "eval/results/winogavil" &

  while [ "$(jobs -rp | wc -l)" -ge 1 ]; do
    wait -n
  done
done

wait
