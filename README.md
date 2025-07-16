# EVIT-SAM Agent

This agent uses the EfficientViT SAM model by MIT's Han Lab for image segmentation and analysis.
It can process images and provide segmentation results through the Agentverse.

Default SAM Model Parameters:

```
    image_data: bytes,
    mime_type: str,
    points_per_side: int = 64,
    pred_iou_thresh: float = 0.8,
    stability_score_thresh: float = 0.85,
    crop_n_layers: int = 0,
    crop_n_points_downscale_factor: int = 2,
    min_mask_region_area: int = 0
```

Example Query:

“Segment this image please.”
[Attach an image]

Example Response:

"Your image has been processed with EfficientViT SAM:"
[Image attached]
