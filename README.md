# EVIT-SAM Agent

This agent uses the EfficientViT SAM model by MIT's Han Lab for image segmentation and analysis.
It can process images and provide segmentation results through the Agentverse.

---

**Default SAM Model Parameters:**

```
Guiding points per batch (default=64)

Prediction IOU threshold (default=0.8)

Stability score threshold (default=0.85)

Box NMS threshold (default=0.7)
```

**Example Query:**

“Segment this image please.”
[Attach an image]

**Example Response:**

"Your image has been processed with EfficientViT SAM:"
[Image attached]

---

Open source model available here: https://evitsam.hanlab.ai/
