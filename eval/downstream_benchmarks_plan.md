# Downstream Benchmark Integration Plan

## Goal

Add the rebuttal downstream benchmarks into `eval/`, verify the code path is correct, then rerun the requested model suite:

- CLIP ViT-L/14@224 baseline plus four trained CLIP-R variants.
- SigLIP So400m/14@384 baseline plus four trained ReasonSigLIP variants.
- Benchmarks: PASCAL VOC, ADE20K, NYUv2 depth, NYUv2 normals, NAVI depth, NAVI normals, RefCOCO, RefCOCO+.

The immediate concern is that previous rebuttal results made CLIP look stronger than SigLIP in some downstream probes. Before rerunning, the evaluation protocol must be audited so a CLIP-vs-SigLIP gap is interpretable rather than caused by an implementation mismatch.

## Current Evidence To Recheck

- Rebuttal scripts exist under `rebuttal/bz/`:
  - `dense_segmentation_probe.py`
  - `nyuv2_geometry_probe.py`
  - `navi_geometry_probe.py`
  - `refcoco_candidate_grounding_eval.py`
- Previous outputs are under `rebuttal/bz/downstream_representation/`.
- Datasets are present locally:
  - `rebuttal/downstream_data/VOCdevkit/VOC2012`
  - `rebuttal/downstream_data/ADEChallengeData2016`
  - `rebuttal/downstream_data/NYUv2`
  - `rebuttal/downstream_data/NAVI/navi_v1.0`
  - `rebuttal/downstream_repos/refer/data`
- The requested Python environment is `/home/localadmin/venvs/llm`.
- Four GPUs are available and may be used.

## Model Matrix

### CLIP L/14@224

Use `openai/clip-vit-large-patch14` as the baseline processor/model family.

| Role | Model |
|---|---|
| baseline | `openai/clip-vit-large-patch14` |
| rea/direct | `fesvhtr/clip-r-rea-run1219-621` |
| des/direct | `fesvhtr/clip-r-des-run0131-949` |
| s1 | `fesvhtr/clip-r-s1-run1207-1280` |
| s2 | `fesvhtr/clip-r-s2-run1219-505` |

### SigLIP So400m/14@384

Use `google/siglip-so400m-patch14-384` as the baseline processor/model family.

| Role | Model |
|---|---|
| baseline | `google/siglip-so400m-patch14-384` |
| rea/direct | `fesvhtr/siglip-r-rea-run0126-1241` |
| des/direct | `fesvhtr/siglip-r-des-run0131-1266` |
| s1 | `fesvhtr/siglip-r-s1-run0201-1280` |
| s2 | `fesvhtr/siglip-r-s2-run0203-673` |

If any Hugging Face snapshot is missing locally, the run script should download it through the requested environment and record that in the log.

## Code Audit Gates Before Full Runs

1. Model loading:
   - Use `AutoModel`, `AutoImageProcessor`, and `AutoTokenizer` explicitly.
   - Use the baseline processor for trained variants when the trained repo lacks processor files.
   - Do not infer CLIP/SigLIP behavior from loose substring checks inside benchmark logic.

2. Image preprocessing:
   - Resize/crop labels with the same geometry as the model image processor.
   - Derive `image_size`, `patch_size`, hidden size, and patch grid from `model.config.vision_config`.
   - Confirm CLIP-L/14@224 uses a 16x16 patch grid and SigLIP So400m/14@384 uses the expected 27x27 grid.

3. Patch-token extraction:
   - Use `model.vision_model(...).last_hidden_state`.
   - Drop a leading CLS token only when token count is not a square.
   - Apply `post_layernorm` when the vision model exposes it.
   - Assert the final token count equals `grid * grid`.

4. Dense probes:
   - Freeze the backbone.
   - Train only the same lightweight linear head.
   - Use identical seed, epochs, optimizer, batch size policy, and metric code across model families unless memory requires a documented smaller batch.
   - Save both JSON metrics and run logs.

5. Grounding:
   - Use official REFER candidate boxes from `instances.json`.
   - Score the same candidate crops for every model.
   - Evaluate expression-level Acc@0.5, exact annotation accuracy, mean IoU, and mean candidate count.
   - Keep RefCOCO and RefCOCO+ split handling explicit (`unc`, `testA` by default).

6. Sanity checks:
   - Run tiny smoke tests for every benchmark with `--max-*` limits.
   - Print model config, processor path, image size, patch size, grid, hidden size, train/val sample counts, and output path.
   - Fail on empty splits, non-square patch tokens, missing labels, or zero valid pixels.

## Expected Results And Interpretation

The strongest correctness expectation is not that SigLIP must beat CLIP on every frozen downstream probe. These probes use lightweight heads over patch tokens and are sensitive to input resolution, patch-grid density, pretraining objective, and dataset domain. However:

- SigLIP So400m baseline should reproduce official SigLIP-paper zero-shot classification numbers within a reasonable tolerance when evaluated with the official-style processor and prompts.
- If official-style classification does not reproduce, downstream CLIP-vs-SigLIP comparisons are not trustworthy yet.
- If official-style classification reproduces but dense probes still favor CLIP, the likely conclusion is that these frozen patch-token probes measure a different transfer property than the original SigLIP classification/retrieval headline metrics.
- If trained SigLIP variants underperform the SigLIP baseline or CLIP variants broadly, inspect processor mismatch, checkpoint compatibility, text preprocessing, overfitting to reasoning data, and whether the trained checkpoint preserved the original projection/vision weights as expected.

Official SigLIP reference values must be collected from the SigLIP or SigLIP 2 paper / official release before the full result table is finalized.

## Implementation Plan

1. Create shared downstream utilities under `eval/`:
   - model spec parsing and loading,
   - patch-feature extraction,
   - CLIP-compatible resize/center-crop label transforms,
   - JSON/Markdown result writing.

2. Add formal eval entrypoints:
   - `eval/eval_downstream_segmentation.py`
   - `eval/eval_downstream_geometry.py`
   - `eval/eval_downstream_grounding.py`
   - `eval/run_downstream_benchmarks.py`

3. Add shell wrapper:
   - `scripts/eval_downstream_benchmarks.sh`
   - Use `/home/localadmin/venvs/llm/bin/python`.
   - Allow `CUDA_VISIBLE_DEVICES=0,1,2,3`.
   - Save outputs under `eval/results/downstream/`.

4. Run smoke tests:
   - one tiny segmentation run,
   - one tiny NYUv2 depth run,
   - one tiny NAVI depth run,
   - one tiny RefCOCO grounding run.

5. Run official baseline reproduction:
   - SigLIP So400m/14@384 zero-shot ImageNet-family check.
   - Compare to official numbers and record the source and tolerance.

6. Run full downstream matrix:
   - 10 models x 8 benchmark tasks.
   - Prefer parallel GPU assignment by independent subprocesses.
   - Keep full logs and write an aggregate CSV/Markdown summary.

7. Analyze anomalies:
   - Compare CLIP vs SigLIP baselines task by task.
   - Compare trained vs baseline within each family.
   - Rerun any suspicious task with a different seed or smaller sample check if the ranking looks unstable.

## Stop Conditions

Do not treat the task as complete until:

- the downstream benchmarks are implemented under `eval/`,
- smoke tests pass in `/home/localadmin/venvs/llm`,
- the requested 10-model matrix has result files for every benchmark,
- the SigLIP official reproduction check is documented,
- and the CLIP-better-than-SigLIP cases are explained by verified evidence rather than assumed to be valid.
