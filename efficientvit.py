from gradio_client import Client, handle_file
import base64
from io import BytesIO
from typing import Any, Tuple, Optional
from PIL import Image

client = Client("https://evitsam.hanlab.ai/")

async def process_image_with_sam(
    image_data: bytes, 
    mime_type: str,
    points_per_side: int = 64,
    pred_iou_thresh: float = 0.8,
    stability_score_thresh: float = 0.85,
    crop_n_layers: int = 0,
    crop_n_points_downscale_factor: int = 2,
    min_mask_region_area: int = 0
) -> Tuple[Optional[bytes], Optional[str]]:
    """
    Process an image using the EfficientViT SAM model.
    
    Args:
        image_data: Raw image bytes
        mime_type: MIME type of the image
        points_per_side: Number of points per side for SAM
        pred_iou_thresh: IoU threshold for mask prediction
        stability_score_thresh: Stability score threshold
        crop_n_layers: Number of layers for cropping
        crop_n_points_downscale_factor: Downscale factor for points in cropped layers
        min_mask_region_area: Minimum area for mask regions
    
    Returns:
        Tuple of (segmented_image_bytes, analysis_text)
    """
    try:
        image = Image.open(BytesIO(image_data))
        
        temp_img_path = "temp_image.png"
        image.save(temp_img_path)
        
        result = client.predict(
            param_0=handle_file(temp_img_path),
            param_2=points_per_side,
            param_3=pred_iou_thresh,
            param_4=stability_score_thresh,
            param_5=stability_score_thresh,  # Reusing stability score for simplicity
            api_name="/lambda_3"
        )
        
        segmented_image_path = result
        analysis_text = "Image processed with EfficientViT SAM model"
        
        with open(segmented_image_path, "rb") as f:
            segmented_image_bytes = f.read()
            
        return segmented_image_bytes, analysis_text
        
    except Exception as e:
        print(f"Error processing image with SAM: {str(e)}")
        return None, str(e)

async def get_image(
    content: list[dict[str, Any]], 
    tool: dict[str, Any] | None = None
) -> Tuple[Optional[bytes], Optional[str]]:
    """
    Process image content using the SAM model.
    
    Args:
        content: List of content items (should contain image data)
        tool: Optional tool configuration
        
    Returns:
        Tuple of (segmented_image_bytes, analysis_text)
    """
    for item in content:
        if item.get("type") == "resource":
            mime_type = item.get("mime_type", "")
            if mime_type.startswith("image/"):
                try:
                    image_data = base64.b64decode(item["contents"])
                    return await process_image_with_sam(image_data, mime_type)
                except Exception as e:
                    return None, f"Error processing image: {str(e)}"
            else:
                return None, f"Unsupported mime type: {mime_type}"
    
    return None, "No valid image content found"