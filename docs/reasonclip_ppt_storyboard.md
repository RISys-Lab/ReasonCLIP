# ReasonCLIP PPT Storyboard

Audience: non-specialist technical audience, research group visitors, potential collaborators, and reviewers who need to quickly understand why ReasonCLIP matters.

Tone: clear, confident, visual, and scale-oriented. Avoid dense equations except as optional speaker notes. Emphasize the problem, data scale, experiment scale, and application value.

Visual system:
- Primary colors: ReasonCLIP blue `#5675A2`, ReasonCLIP purple `#722E5F`.
- Soft backgrounds: light blue `#E8EDF8`, light purple `#FBF3FA`, warm off-white `#FAF8F5`.
- Supporting category colors from the paper figures: muted red, mustard yellow, green, teal, slate blue.
- Brand reference: `asset/reasonclip_wordmark_compact.svg`.
- Paper figure references rendered in `tmp/ppt_fig_refs/`: `teaser_v2.png`, `dataset.png`, `method.png`, `sample_main.png`, `app_stat.png`.

## Slide 1 - ReasonCLIP-58M

Main message:
ReasonCLIP turns CLIP from "matching images with descriptions" into a visual encoder that also understands visually grounded commonsense reasoning.

On-slide content:
- ReasonCLIP-58M
- Visually Grounded Commonsense Reasoning Supervision for CLIP
- 58.9M reasoning supervision samples
- No architecture change. No extra inference cost.

Speaker angle:
CLIP is everywhere in modern multimodal AI, but it mostly learns descriptive matching. ReasonCLIP asks a simple question: can we make the same kind of visual encoder reason better by changing the supervision rather than changing the architecture?

Visual direction:
Hero cover with the ReasonCLIP wordmark style, soft blue-purple neural curves, small image/text embedding nodes, and a clear "description -> reasoning" transformation.

Image prompt:
Create a polished 16:9 presentation cover for "ReasonCLIP-58M" about visually grounded commonsense reasoning. Use a clean research-lab aesthetic, soft off-white background, blue-purple gradient lines, sparse embedding nodes, and a subtle transformation from image-description matching to visual reasoning. Include only these exact large text elements: "ReasonCLIP-58M" and "From Description to Reasoning". No logos, no watermark, no extra small unreadable text.

## Slide 2 - The Problem: CLIP Sees, But Does It Reason?

Main message:
Most CLIP-style models are trained to align images with descriptive captions, but real multimodal systems need visual grounding, evidence awareness, and commonsense reasoning.

On-slide content:
- CLIP training is dominated by descriptive image-text alignment.
- Downstream systems need more than "what is in the image".
- They need to infer "why", "how", "where", "what happens next", and "what is plausible".
- The gap: descriptive pretraining vs reasoning-oriented applications.

Speaker angle:
For retrieval, "two cats stand near monitors" may be enough. For an assistant, that is not enough; it should notice the cats are staring at a mouse on a monitor and infer hunting behavior. The missing piece is not more text length alone, but reasoning that remains grounded in visible evidence.

Visual direction:
Split scene: left shows a plain image-caption match, right shows highlighted visual evidence leading to a reasoning conclusion. Avoid academic clutter.

Image prompt:
Create a 16:9 slide visual explaining the limitation of descriptive image-text matching. Left side: a simple image-card connected to a caption card labeled "Description". Right side: the same image-card with highlighted visual evidence connected to a reasoning card labeled "Grounded Reasoning". Use soft blue and purple accents, minimal icons, high-end conference presentation style. Text must be limited to "Description", "Visual Evidence", and "Grounded Reasoning".

## Slide 3 - Our Answer: 58.9M Reasoning Supervision Samples

Main message:
ReasonCLIP builds a large reasoning-oriented supervision pipeline from CC12M: refined descriptions, open-form reasoning, and category-specific reasoning.

On-slide content:
- 10.4M valid CC12M images retained.
- 31.2M refined descriptive image-text pairs.
- ReasonLite-42M: open-form visually verifiable reasoning.
- ReasonPro-16M: category-specific reasoning across five types.
- Total: 58.9M visually grounded commonsense reasoning samples.

Speaker angle:
The important point is scale plus control. We are not hand-writing a small benchmark; we are building a large training signal that can move representation learning, while still constraining reasoning to what can be visually verified.

Visual direction:
Data funnel or pipeline: CC12M images -> refined captions -> ReasonLite + ReasonPro -> ReasonCLIP-58M. Use large numeric cards and avoid tiny table text.

Image prompt:
Create a 16:9 presentation infographic showing a large-scale data pipeline. Flow: "10.4M images" to "31.2M refined captions" branching into "ReasonLite-42M" and "ReasonPro-16M", then merging into "ReasonCLIP-58M". Use rounded data cards, blue for ReasonLite, purple for ReasonPro, off-white background, subtle dataset thumbnails as abstract blocks. Keep text exactly as listed and readable.

## Slide 4 - Two Complementary Datasets

Main message:
ReasonLite teaches broad reasoning awareness; ReasonPro teaches structured reasoning categories.

On-slide content:
- ReasonLite-42M: open-form reasoning from visual evidence.
- 4.67M images, 42.0M triplet pairs.
- ReasonPro-16M: structured category-specific supervision.
- 5.52M valid images, 16.56M image-reasoning pairs.
- Five categories: spatial/geometric, attribute/state, creature/action, temporal/phase, physical intuition.

Speaker angle:
ReasonLite is the broad exposure stage: many visual scenes, many grounded reasoning statements. ReasonPro is the organizing stage: it teaches the model that different types of reasoning are distinct patterns rather than one generic caption style.

Visual direction:
Two asymmetric panels, not identical cards. Left: organic cloud of open-form reasoning snippets. Right: five-category wheel or compass.

Image prompt:
Create a 16:9 slide comparing two datasets without using a table. Left side: "ReasonLite-42M" as a broad blue cloud of visual evidence snippets. Right side: "ReasonPro-16M" as a purple five-segment reasoning wheel with labels "Spatial", "Attribute", "Action", "Temporal", "Physics". Include numeric badges "42.0M pairs" and "16.56M pairs". Elegant research style, readable text, off-white background.

## Slide 5 - Quality Control at Scale

Main message:
Large data is useful only if systematic errors are controlled. ReasonCLIP uses manual pilot validation, prompt refinement, and automatic filtering.

On-slide content:
- Pilot validation: 500 samples per stage.
- Five graduate reviewers independently checked generated samples.
- Large-scale generation starts only after pass rate > 99.5%.
- Final random inspection remains > 99.0%.
- Filtering removes hallucination, false causality, over-extension, malformed or degenerate captions.

Speaker angle:
For non-specialists, this slide answers the natural concern: if the data is generated by models, why should we trust it? The answer is that we constrain the reasoning level, filter bad patterns, and validate before and after generation.

Visual direction:
Quality gate sequence: Generate -> Review -> Refine -> Filter -> Release. Use checkmarks, not dense text.

Image prompt:
Create a 16:9 presentation visual for quality control in a large AI dataset. Show a clean horizontal pipeline with five gates: "Generate", "Review", "Refine", "Filter", "Release". Add two large quality badges: ">99.5% pilot pass" and ">99.0% final pass". Use blue-purple palette with small red warning markers for "hallucination" and "false causality". Minimal, polished, readable.

## Slide 6 - Training: Add Reasoning Without Rebuilding CLIP

Main message:
ReasonCLIP uses staged continual pretraining to add reasoning while preserving the original CLIP/SigLIP architecture and inference pipeline.

On-slide content:
- Stage 0: baseline descriptive or naive reasoning pretraining.
- Stage 1: reasoning-aware alignment with ReasonLite.
- Stage 2: explicit category-level reasoning with ReasonPro.
- Stage 3: drop-in visual encoder for MLLMs.
- Classification heads are used only during training and discarded at inference.

Speaker angle:
The design is practical. We are not asking people to redesign their multimodal pipeline. We train the visual encoder better, then use it as a drop-in replacement.

Visual direction:
Layered staircase or route map. Make Stage 1 and Stage 2 visually distinct: blue "alignment preservation", purple "reasoning organization".

Image prompt:
Create a 16:9 slide visual showing a four-stage training route for ReasonCLIP. Use a clean staircase or route-map composition with stages: "Stage 0 Baseline", "Stage 1 Reasoning-Aware Alignment", "Stage 2 Category Reasoning", "Stage 3 Drop-in Integration". Blue highlights for Stage 1, purple for Stage 2, subtle icon of a visual encoder moving into an MLLM pipeline. No equations, no tiny text.

## Slide 7 - Experimental Scale

Main message:
ReasonCLIP is validated at meaningful scale across architectures, model sizes, retrieval tasks, reasoning tasks, and MLLM integration.

On-slide content:
- Data generation: about 3.8k A100-64GB GPU hours.
- Model training: about 3.5k A100-64GB GPU hours.
- Six CLIP and SigLIP variants across scales.
- Effective batch size up to 32,768.
- LLaVA-NeXT integration: 558K pretraining samples and 779K fine-tuning samples.

Speaker angle:
This is not a small proof-of-concept. The experiments are large enough to test whether the idea transfers across model families and downstream settings.

Visual direction:
Dashboard style: GPU-hours, model variants, batch size, MLLM samples. Use large numerals and small captions.

Image prompt:
Create a 16:9 data-dashboard slide for an AI research project. Show four large metric tiles: "3.8k GPU hours data generation", "3.5k GPU hours training", "6 CLIP/SigLIP variants", "32,768 max effective batch size". Include a smaller footer strip: "LLaVA-NeXT: 558K pretrain + 779K fine-tune". Use clean typography, blue-purple gradient accents, off-white background, no charts with tiny labels.

## Slide 8 - RCLIP-Bench: Finding Where Reasoning Fails

Main message:
RCLIP-Bench diagnoses three levels of visually grounded reasoning instead of treating all image-text matching as one task.

On-slide content:
- V1 Visual Grounding: is the caption factually consistent with the image?
- V2 Evidence Awareness: can the model reject reasoning built on wrong facts?
- V3 Visual Reasoning: can it reject wrong reasoning from correct facts?
- 5,000 images per version, 125,000 annotations per version.

Speaker angle:
The benchmark matters because a model can look good on standard retrieval while still failing to reason. RCLIP-Bench separates perception errors from reasoning errors.

Visual direction:
Three-tier diagnostic ladder. Each tier shows a simple example: fact error, wrong evidence, wrong conclusion.

Image prompt:
Create a 16:9 diagnostic benchmark slide with a three-level ladder. Levels: "V1 Visual Grounding", "V2 Evidence Awareness", "V3 Visual Reasoning". Add a large badge "125K annotations per version" and "5K images". Use muted red/yellow/slate level colors like the paper figure, with blue-purple ReasonCLIP accents. Make it accessible to non-specialists, no dense paragraphs.

## Slide 9 - Results: Better Retrieval and Better Reasoning

Main message:
Reasoning-oriented supervision improves not only reasoning benchmarks but also zero-shot retrieval, showing that CLIP's representation space can be extended rather than broken.

On-slide content:
- Consistent gains over CLIP/SigLIP baselines across model sizes.
- COCO, Flickr, Urban1K, and RCLIP-V3 retrieval improve under the same backbone family.
- Example: CLIP-L/14-336 Urban1K I->T R@1 improves from 73.0 to 83.0 after Stage 1.
- Example: RCLIP-V3 T->I R@1 improves from 33.2 to 42.8 after Stage 2.

Speaker angle:
The surprising part is that adding reasoning supervision does not simply trade off retrieval quality. It can improve standard retrieval and reasoning retrieval at the same time.

Visual direction:
Before/after slope chart with two highlighted examples. Keep the full table out of the slide.

Image prompt:
Create a 16:9 results slide with two clean before-after slope charts. Chart 1: "Urban1K I->T R@1" from "73.0 CLIP" to "83.0 ReasonCLIP". Chart 2: "RCLIP-V3 T->I R@1" from "33.2 CLIP" to "42.8 ReasonCLIP". Use blue for baseline, purple for ReasonCLIP, include upward arrows, minimal labels, conference presentation style.

## Slide 10 - Results: Stronger Compositional Reasoning

Main message:
ReasonCLIP improves structured compositional reasoning across benchmarks without task-specific fine-tuning.

On-slide content:
- Evaluated on WhatsUp, VALSE, CREPE, SugarCREPE, and SugarCrepe++.
- Average gains are roughly +5 to +7 points across CLIP-scale variants.
- Example: CLIP-L/14-336 average improves from 51.8 to 58.1 (+7.7).
- ReasonCLIP can also strengthen specialized methods such as READ-CLIP.

Speaker angle:
Compositional reasoning is where models must understand relationships, attributes, and structure, not just objects. This is close to what real users need when asking about visual scenes.

Visual direction:
Puzzle/relationship metaphor: objects, arrows, attributes, spatial relations. Include a compact gain badge.

Image prompt:
Create a 16:9 slide about compositional reasoning gains. Use a visual puzzle motif with objects, attributes, arrows, and spatial relations snapping together. Include a prominent metric badge: "+7.7 avg points on CLIP-L/14-336". Add small benchmark chips: "WhatsUp", "VALSE", "CREPE", "SugarCREPE", "SugarCrepe++". Blue-purple palette, polished and not cluttered.

## Slide 11 - Application Value: A Better Visual Tower for MLLMs

Main message:
ReasonCLIP can be used as a drop-in visual encoder in an MLLM, improving reasoning-oriented downstream performance without adding inference cost.

On-slide content:
- Integrated into LLaVA-NeXT with Qwen3-1.7B language backbone.
- Same training setup, frozen vision tower.
- Improves OKVQA, GQA, MME, MMStar, MMVP, and other benchmarks.
- No additional inference cost from the visual encoder replacement.

Speaker angle:
This is the application story. If a system already uses a CLIP-like visual tower, ReasonCLIP is a practical upgrade path: same role, stronger reasoning-aware representations.

Visual direction:
MLLM pipeline: image -> ReasonCLIP visual tower -> language model -> better answer. Add "drop-in" badge.

Image prompt:
Create a 16:9 application slide showing ReasonCLIP as a drop-in visual tower for a multimodal assistant. Flow: image input -> "ReasonCLIP visual tower" -> "LLaVA-NeXT / MLLM" -> answer card. Include badges "drop-in replacement" and "no extra inference cost". Use friendly but professional AI product style, blue-purple palette, clean iconography.

## Slide 12 - Takeaway

Main message:
ReasonCLIP shows that visually grounded reasoning can be injected into CLIP-style encoders through data and training, not by changing the architecture.

On-slide content:
- Problem: CLIP is descriptive, but applications need reasoning.
- Data: 58.9M visually grounded reasoning samples.
- Method: two-stage continual pretraining.
- Evidence: gains in retrieval, reasoning, compositionality, and MLLM integration.
- Value: practical path toward reasoning-aware vision-language foundation models.

Speaker angle:
Close with the big idea: reasoning ability can begin at the visual representation level. Better visual encoders make downstream multimodal systems more capable before the language model even starts answering.

Visual direction:
Summary constellation: five nodes connected around ReasonCLIP - Problem, Data, Method, Results, Application. End with a clean, memorable visual.

Image prompt:
Create a 16:9 closing slide for ReasonCLIP. Center node "ReasonCLIP" connected to five nodes: "Problem", "Data", "Training", "Results", "Applications". Use elegant blue-purple gradient network lines, off-white background, subtle glow, professional conference keynote style. Include a final line: "Reasoning starts in the visual representation." Keep all text readable and minimal.

## Optional Appendix Slides

### Appendix A - Dataset Construction Details

Use only if audience asks about data trust or reproducibility.

Content:
- CC12M source and URL-based usage.
- Qwen2.5-VL-72B for CC12M-Refined and ReasonLite.
- Qwen3-VL-32B for ReasonPro.
- Rule-based filtering and post-hoc inspection.

### Appendix B - Full Evaluation Coverage

Use only if audience is technical.

Content:
- Retrieval: COCO, Flickr30K, Urban1K, RCLIP-V3.
- Commonsense reasoning: WinoGAViL, RCLIP-Bench V1/V2/V3.
- Compositional reasoning: WhatsUp, VALSE, CREPE, SugarCREPE, SugarCrepe++.
- MLLM: AI2D, ChartQA, SciQA, RealWorldQA, VisualLogic, OKVQA, GQA, MME, MMStar, MMVP.

## Evidence Notes From Paper

- Abstract: ReasonCLIP-58M integrates large-scale reasoning supervision into CLIP-style models without architecture changes.
- Main data scale: 58.9M visually grounded commonsense reasoning samples.
- CC12M-Refined: 10,388,539 images and 31,165,584 image-text pairs.
- ReasonLite: 4,668,515 images and 42,016,635 triplet pairs.
- ReasonPro: 5,521,563 valid images and 16,564,689 image-reasoning pairs.
- ReasonPro category counts: 4.77M spatial/geometric, 4.88M attribute/state, 2.47M creature/action, 0.95M temporal/phase, 3.50M physical intuition.
- RCLIP-Bench: V1/V2/V3, 5,000 images per version, 125,000 annotations per version.
- Quality control: pilot validation uses 500 samples per stage and five graduate reviewers; large-scale generation starts after >99.5% pass; final random inspection remains >99.0%.
- Training scale: about 3.8k A100-64GB GPU hours for data generation and 3.5k A100-64GB GPU hours for model training.
- Model coverage: six CLIP/SigLIP variants across model scales.
- Training batches: effective batch size up to 32,768.
- MLLM integration: LLaVA-NeXT with Qwen3-1.7B, 558K pretraining samples and 779K fine-tuning samples.
