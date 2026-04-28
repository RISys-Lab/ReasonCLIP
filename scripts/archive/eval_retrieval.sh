source "/home/localadmin/bz/CLIP-R/model/models.sh"
if [ "${#models[@]}" -ne "${#processors[@]}" ]; then
  echo "models/processors length mismatch: ${#models[@]} vs ${#processors[@]}"
  exit 1
fi

for i in "${!models[@]}"; do
  python eval/eval_retrieval.py \
    --model_path "${models[$i]}" \
    --processor_path "${processors[$i]}" \
    --model_name auto \
    --dataset_name urban1k \
    --split test \
    --batch_size 512 \
    --device cuda:0 \
    --skip_if_exists \
    --results_dir "$WORK/fmohamma/CLIP-R/eval/results/retrieval_urban1k" &

  while [ "$(jobs -rp | wc -l)" -ge 1 ]; do
    wait -n
  done
done


# for i in "${!models[@]}"; do
#   python eval/eval_retrieval.py \
#     --model_path "${models[$i]}" \
#     --processor_path "${processors[$i]}" \
#     --model_name auto \
#     --dataset_name mscoco \
#     --split test \
#     --coco_captions_json "$COCO2017_CAPTIONS_JSON" \
#     --local_image_dir "$COCO2017_IMAGE_DIR" \
#     --batch_size 512 \
#     --device cuda:0 \
#     --skip_if_exists \
#     --results_dir "$WORK/fmohamma/CLIP-R/eval/results/retrieval_coco2017" &

#   while [ "$(jobs -rp | wc -l)" -ge 1 ]; do
#     wait -n
#   done
# done

for i in "${!models[@]}"; do
  python eval/eval_retrieval.py \
    --model_path "${models[$i]}" \
    --processor_path "${processors[$i]}" \
    --model_name auto \
    --dataset_name wds_mscoco \
    --split test \
    --batch_size 512 \
    --device cuda:0 \
    --skip_if_exists \
    --results_dir "$WORK/fmohamma/CLIP-R/eval/results/retrieval_wds_mscoco" &

  while [ "$(jobs -rp | wc -l)" -ge 1 ]; do
    wait -n
  done
done

for i in "${!models[@]}"; do
  python eval/eval_retrieval.py \
    --model_path "${models[$i]}" \
    --processor_path "${processors[$i]}" \
    --model_name auto \
    --dataset_name flickr30k \
    --split test \
    --batch_size 512 \
    --device cuda:0 \
    --skip_if_exists \
    --results_dir "$WORK/fmohamma/CLIP-R/eval/results/retrieval_flickr30k" &

  while [ "$(jobs -rp | wc -l)" -ge 1 ]; do
    wait -n
  done
done




wait
