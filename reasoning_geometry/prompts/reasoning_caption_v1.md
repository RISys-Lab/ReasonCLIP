# Reasoning Caption Annotation Prompt v1

Given an image and a detailed human-written caption, write 1-3 short visually grounded commonsense reasoning captions.

Each caption must:

- contain visible evidence from the image/caption,
- state a commonsense conclusion that follows from that evidence,
- stay grounded in the visible scene,
- avoid private mental states, unverifiable causes, or broad stories.

Output schema:

```json
{
  "reasoning_captions": [
    {
      "caption": "... because ...",
      "visible_evidence": "...",
      "commonsense_conclusion": "...",
      "reasoning_type": "support|containment|protection|use_or_function|state_or_activity|spatial_relation|material_state|affordance|other"
    }
  ]
}
```
