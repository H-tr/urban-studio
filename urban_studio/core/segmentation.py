from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from urban_studio.utils.segmentation_utils import ade_palette

processor = SegformerImageProcessor.from_pretrained(
    "nvidia/segformer-b0-finetuned-ade-512-512"
)
model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b0-finetuned-ade-512-512"
)


def seg_image(image):
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits  # shape (batch_size, num_labels, height/4, width/4)
    # First, rescale logits to original image size
    logits = nn.functional.interpolate(
        logits.detach().cpu(),
        size=image.size[::-1],  # (height, width)
        mode="bilinear",
        align_corners=False,
    )

    seg = logits.argmax(dim=1)[0]
    color_seg = np.zeros(
        (seg.shape[0], seg.shape[1], 3), dtype=np.uint8
    )  # height, width, 3
    palette = np.array(ade_palette())
    for label, color in enumerate(palette):
        color_seg[seg == label, :] = color
    # Convert to BGR
    color_seg = color_seg[..., ::-1]

    # Show image + mask
    img = np.array(image) * 0.5 + color_seg * 0.5
    img = img.astype(np.uint8)

    return img
