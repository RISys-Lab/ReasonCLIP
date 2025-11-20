from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-72B-Instruct")

system_prompt = """
You are a reasoning agent specialized in egocentric hand-object interaction understanding. 
Your task is to analyze images captured from a first-person perspective and generate a caption 
that describes all hand and object interaction details relevant to 3D hand reconstruction.

Given a single input image, output one concise sentence that specifies which hand(s) 
(left/right/both) are visible, their pose, and their interaction with any objects. 
Be precise and descriptive about the hand-object interaction without adding explanations or speculations.
"""

user_prompt = """
Both hands are interacting with a mannequin, complicating the reconstruction due to complex hand-object contact.
"""

print("System prompt tokens:", len(tokenizer.tokenize(system_prompt)))

print("User prompt tokens:", len(tokenizer.tokenize(user_prompt)))
