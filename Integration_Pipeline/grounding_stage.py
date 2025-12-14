import os
import torch
import numpy as np
from PIL import Image
from PIL import ImageDraw
from groundingdino.util.slconfig import SLConfig
from groundingdino.models import build_model
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
import groundingdino.datasets.transforms as T


def load_model(config_path, checkpoint_path, device="cuda"):
    args = SLConfig.fromfile(config_path)
    args.device = device
    model = build_model(args)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    return model.to(device)


def load_image(image_path):
    image_pil = Image.open(image_path).convert("RGB")

    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]),
    ])

    image_tensor, _ = transform(image_pil, None)
    return image_pil, image_tensor


def get_boxes(model, image_tensor, caption,
              box_threshold=0.5, text_threshold=0.4,
              device="cuda"):

    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption += "."

    image_tensor = image_tensor.to(device)
    model = model.to(device)

    with torch.no_grad():
        outputs = model(image_tensor[None], captions=[caption])

    logits = outputs["pred_logits"].sigmoid()[0]      # (nq, 256)
    boxes = outputs["pred_boxes"][0]                  # (nq, 4)

    # filtering by box confidence
    mask = logits.max(dim=1)[0] > box_threshold
    logits_filt = logits[mask]
    boxes_filt = boxes[mask]

    # decode phrases
    tokenizer = model.tokenizer
    tokenized = tokenizer(caption)
    phrases = [
        get_phrases_from_posmap(l > text_threshold, tokenized, tokenizer)
        for l in logits_filt
    ]

    scores = logits_filt.max(dim=1)[0].cpu()
    return boxes_filt.cpu(), phrases, scores


def visualize(image_pil, boxes, phrases, out_path):
    W, H = image_pil.size
    draw = ImageDraw.Draw(image_pil)

    for box, phrase in zip(boxes, phrases):
        # convert normalized cxcywh → absolute xyxy
        cx, cy, w, h = box.tolist()
        x0 = (cx - w / 2) * W
        y0 = (cy - h / 2) * H
        x1 = (cx + w / 2) * W
        y1 = (cy + h / 2) * H

        color = tuple(np.random.randint(0, 255, 3).tolist())
        draw.rectangle([x0, y0, x1, y1], outline=color, width=4)
        draw.text((x0, y0), phrase, fill=color)

    image_pil.save(out_path)


if __name__ == "__main__":
    # ---- UPDATE THESE ----
    CONFIG = "/home/jay/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    CHECKPOINT = "/home/jay/GroundingDINO/weights/groundingdino_swint_ogc.pth"

    IMAGE = "/home/jay/Language Aware Manipulation/Pybullet_test/captured/front.png"
    PROMPT = "red sphere"

    # ----------------------
    model = load_model(CONFIG, CHECKPOINT)
    image_pil, image_tensor = load_image(IMAGE)
    boxes, phrases = get_boxes(model, image_tensor, PROMPT)

    print("Detections:")
    for b, p in zip(boxes, phrases):
        print(p, b.tolist())

    visualize(image_pil, boxes, phrases, "outputpb.png")
    print("Saved: outputpb.png")
