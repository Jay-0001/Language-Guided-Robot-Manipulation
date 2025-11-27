from transformers import BlipProcessor, BlipForQuestionAnswering
from PIL import Image
import matplotlib.pyplot as plt
import torch

# 1️⃣ Load model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(device)

# 2️⃣ Load your CLEVR-Ref+ sample image
image_path = "CLEVR_img.jpg"
image = Image.open(image_path).convert("RGB")

# 3️⃣ Define an intricate reasoning query
# Try queries involving attributes, relations, or context
query = "Which object is metallic and placed inside another object?"

# 4️⃣ Forward pass
inputs = processor(image, query, return_tensors="pt").to(device)
out = model.generate(**inputs, max_new_tokens=20)
answer = processor.decode(out[0], skip_special_tokens=True)

# 5️⃣ Display image + query + answer
plt.figure(figsize=(8,6))
plt.imshow(image)
plt.axis("off")
plt.title(f"Query: {query}\nAnswer: {answer}", fontsize=12)
plt.tight_layout()
plt.show()

print(f"Query: {query}")
print(f"BLIP answer: {answer}")
