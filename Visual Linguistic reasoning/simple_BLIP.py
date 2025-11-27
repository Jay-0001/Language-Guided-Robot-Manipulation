from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import matplotlib.pyplot as plt
import re

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model & processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# Helper to extract (action, object)
def extract_action_object(text):
    verbs = re.findall(r'\b(push|pick|move|lift|grasp|pull|place|drop)\b', text.lower())
    action = verbs[0] if verbs else "unknown"
    obj_match = re.search(rf"{action}\s+([a-z\s]+)", text.lower()) if action != "unknown" else None
    obj = obj_match.group(1).strip() if obj_match else "unknown"
    return action, obj

# Run BLIP reasoning
def blip_action_object(image_path, command):
    image = Image.open(image_path).convert("RGB")
    prompt = f"In this image, the robot should {command}. Describe the object being interacted with and the action involved."
    inputs = processor(image, prompt, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=30)
    caption = processor.decode(out[0], skip_special_tokens=True)
    action, obj = extract_action_object(caption)
    return caption, action, obj, image

# Example
command = "grasp the green sphere"
path = r'W:\Binghamton\Semester 2\Robot Perception\Project\Implementations\Dataset\rlbench_simple_dataset\Images\0000.png'
caption, action, obj, image = blip_action_object(path, command)

# Display
plt.figure(figsize=(8, 6))
plt.imshow(image)
plt.axis("off")
plt.title("BLIP Vision–Language Reasoning", fontsize=14, pad=15)

text = (
    f"Command: {command}\n"
    f"BLIP Description: {caption}\n"
    f"Extracted Action: {action}\n"
    f"Extracted Object: {obj}"
)

plt.gcf().text(0.02, 0.02, text, fontsize=10, va="bottom", ha="left",
               bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8))
plt.tight_layout()
plt.show()
