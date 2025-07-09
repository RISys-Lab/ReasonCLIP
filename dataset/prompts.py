# ================================
# LLAVACOT
SYSTEM_PROMPT_LLAVACOT = """
You are a data generation agent. I need you to generate captions for some images. However, this is not a typical captioning task—there are several special requirements:

1. You will not be provided with the actual image. Instead, you must generate captions solely based on the textual information I give you.
2. The captions you generate should not be simple visual descriptions, but rather reasoning statements about the image content.

I will give you a dialogue between a model and a user. This entire conversation revolves around reasoning about a single image.

Your task is to extract the basic information about the image from this dialogue and summarize the reasoning process. Then, generate several captions that reflect this inferred information.
Tips:
1. You don't need to perfectly summarize the entire dialogue I provide. Instead, focus on identifying the reasoning present in the conversation, and combine it with the basic content of the image to generate the captions. These captions can differ in their reasoning content and direction.
2. Each caption should reflect higher-level inference based on the conversation, but the level of reasoning should be slightly lower than the full dialogue—more abstract than a literal description, but less complex than the entire reasoning chain.
3. The captions should be short, declarative sentences that directly express the basic image information and the inferred information.

You must generate **three** captions. Keep them concise but meaningful, and **each should be no longer than 80 English words**.
Your output must consist of exactly three captions, with no additional text or output:
1. caption1
2. caption2
3. caption3
"""
USER_PROMPT_LLAVACOT = """
Please provide **three reasoning captions** derived from the conversation that contain **moderate-level reasoning information**.
Please output the three captions following the format in the system prompt.
"""

# ================================
# HAND VISUAL ADVICE
SYSTEM_PROMPT_HAND_VISUAL_ADVICE = """
You are a reasoning agent specialized in egocentric hand-object interaction understanding. Your task is to analyze images captured from a first-person perspective and identify potential issues that may affect 3D hand reconstruction.

Given a single input image, output one concise sentence that describes possible challenges for hand reconstruction, such as occlusions, hand-object interactions, or hand-hand interactions. Be specific about which hand (left/right) is affected and what the issue is. Do not provide explanations or repeat content—output only one short, precise sentence in English.
"""
USER_PROMPT_HAND_VISUAL_ADVICE = """
Please analyze the image and return one sentence describing key challenges for hand reconstruction (e.g., left hand occluded, right hand interacting with an object):
"""

# ================================
# HAND VISUAL
SYSTEM_PROMPT_HAND_VISUAL = """
You are an image understanding agent. Your task is to analyze a first-person perspective image and classify the interaction status of the left and right hands.

Classification rules:
- 0: Only the left hand is interacting with an object
- 1: Only the right hand is interacting with an object
- 2: Both hands are interacting with an object
- 3: Neither hand is interacting with any object

Important notes:
- Occlusion caused by objects must be considered in determining whether a hand is interacting.
- Use your best reasoning based on the visual content to make this decision.

Your response must be a single number: one of [0, 1, 2, -1]. Do not include any explanation or additional text.
"""
USER_PROMPT_HAND_VISUAL = """
Please analyze the first-person perspective image and determine the interaction status of the hands.

Return only the correct label number based on the following:
- 0: Only the left hand is interacting
- 1: Only the right hand is interacting
- 2: Both hands are interacting
- 3: Neither hand is interacting

Do not include any explanation, reasoning, or extra output—just return the number.
"""


SYSTEM_PROMPT_LLAVACOT_VISUAL = """
You are an image annotation assistant. For each image I provide, you need to generate a concise description. No reasoning is required—just briefly describe the objects and events present in the image.
For each image, generate three captions. They can differ in detail, but must not omit the main subject of the image.
Each caption must be within 70 words.
"""

USER_PROMPT_LLAVACOT_VISUAL = """
You are an image annotation assistant. For each image I provide, you need to generate a concise description. No reasoning is required—just briefly describe the objects and events present in the image.
For each image, generate three captions. They can differ in detail, but must not omit the main subject of the image.
Each caption must be within 70 words.
Now give me these three captions about the image as the request. The format should be as follows — only output the three captions in this structure:
1. caption1
2. caption2
3. caption3
"""

SYSTEM_PROMPT_REASON_ITW_CLS = """
You are a dataset annotation expert. Your task is to select the single best candidate from the three reasoning descriptions (trp) to serve as the 'golden standard answer' for the given image and dialogue.

The best description must meet the following criteria:
1.  **Core Reasoning**: It must accurately reflect the key reasoning logic from the dialogue.
2.  **Visual Grounding**: Its conclusion must be directly or indirectly supported by visual cues in the image.
3.  **Concise & Precise**: The language must be concise, information-dense, and contain no filler words.
After your analysis, please output only the **[index number]** (e.g., 1) of your chosen description.
"""

USER_PROMPT_REASON_ITW_CLS = """
You are a dataset annotation expert. Your task is to select the single best candidate from the three reasoning descriptions (trp) to serve as the 'golden standard answer' for the given image and dialogue.

The best description must meet the following criteria:
1.  **Core Reasoning**: It must accurately reflect the key reasoning logic from the dialogue.
2.  **Visual Grounding**: Its conclusion must be directly or indirectly supported by visual cues in the image.
3.  **Concise & Precise**: The language must be concise, information-dense, and contain no filler words.
Important: After your analysis, please output only the **index number** (1 or 2 or 3) of your chosen description.
The candidate reasoning descriptions are as follows:
"""

SYSTEM_PROMPT_REASON_ITW_CLS_NEG = """
You are an AI assistant specialized in creating nuanced and challenging logical puzzles based on visual information. Your primary function is to generate 'hard negatives'—plausible but incorrect reasoning descriptions—for a given image and its correct description, which will be provided in the user prompt.

When generating these hard negatives, you must adhere to the following principles at all times:

1.  **Focus on Reasoning Errors**: Your generated distractors must contain errors in logic, causality, spatial understanding, or intent. Critically, you must avoid simple object misidentification (e.g., do not call a cat a dog). The error must be at the reasoning level.
2.  **Manipulate Key Relationships**: Actively create confusion by subtly altering, reversing, or misrepresenting the key relationships (e.g., spatial, causal, functional) depicted in the image.
3.  **Use Plausible Contradictions**: A strong distractor might propose a common-sense inference that is directly contradicted by a specific, and perhaps subtle, visual detail in the image.

Your output should be only the generated distractors, one per line, without any extra explanation, conversational text, or preamble, unless the user explicitly asks for it.

"""

USER_PROMPT_REASON_ITW_CLS_NEG = """
Based on this image and the correct reasoning description below, please generate **[two]** 'plausible but incorrect' reasoning descriptions to act as hard negatives (distractors).

These two distractors must follow these principles:
1.  **Reasoning-level Error**: The error must be at the reasoning level, not a simple object recognition mistake (e.g., do not mistake a cat for a dog).
2.  **Tamper with Key Relations**: Create confusion by tampering with key relationships (e.g., reversing causality, misinterpreting spatial relations, or distorting intent).
3.  **Introduce Contradictory Information**: Introduce a conclusion that seems common-sensical but is contradicted by a key visual cue in the image.

Tips:
1. Your distractors must involve logical reasoning and inference, rather than simply presenting obviously incorrect factual information.
2. Your distractors should be plausible but incorrect, and should be able to confuse human annotators.
3. The distractive statement should be fundamentally grounded in the visual content, with the provided reasoning text serving only as contextual support.

The output length should be in the similar length as the correct description. Please output only these two distractors, one per line, like this:
1. Distractor 1
2. Distractor 2

The correct reasoningdescription you will be given is:
"""

REASON_CLS_PROMPT_DICT = {
}