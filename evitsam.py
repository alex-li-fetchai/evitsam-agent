import base64
import io
from typing import Tuple, Optional, Dict, Any
from gradio_client import Client, handle_file
import tempfile
import os
import re

# Default SAM parameters based on the actual API
DEFAULT_PARAMS = {
    "points_per_side": 64,      # param_2: Guiding points per batch
    "iou_threshold": 0.8,       # param_3: Prediction IOU threshold
    "stability_threshold": 0.85, # param_4: Stability score threshold
    "box_nms_threshold": 0.7    # param_5: Box NMS threshold
}

def parse_prompt_for_parameters(prompt: str) -> Dict[str, Any]:
    """
    Parse natural language prompt to extract parameter updates.
    Returns a dictionary of parameter names and their new values.
    """
    param_updates = {}
    prompt = prompt.lower()
    
    # Parameter mappings
    param_aliases = {
        "points": "points_per_side",
        "iou": "iou_threshold",
        "stability": "stability_threshold",
        "nms": "box_nms_threshold",
        "quality": "iou_threshold"  # Alias for quality
    }
    
    # Look for parameter updates in the prompt
    for alias, param in param_aliases.items():
        # Look for patterns like "set points to 64" or "points=64"
        patterns = [
            fr"{alias}\s*[=:]\s*([\d.]+)",      # points=64 or iou: 0.8
            fr"set\s+{alias}\s+to\s+([\d.]+)",  # set points to 64
            fr"use\s+([\d.]+)\s+for\s+{alias}", # use 64 for points
            fr"{alias}\s+([\d.]+)"              # points 64
        ]
        
        for pattern in patterns:
            match = re.search(pattern, prompt)
            if match:
                try:
                    value = float(match.group(1))
                    # Convert to int if it's a whole number and the parameter should be an int
                    if param == "points_per_side" and value.is_integer():
                        value = int(value)
                    param_updates[param] = value
                    break
                except (ValueError, IndexError):
                    continue
    
    # Handle special cases
    if "higher quality" in prompt:
        param_updates.update({
            "points_per_side": min(128, DEFAULT_PARAMS["points_per_side"] * 2),
            "iou_threshold": min(0.95, DEFAULT_PARAMS["iou_threshold"] + 0.05),
            "stability_threshold": min(0.95, DEFAULT_PARAMS["stability_threshold"] + 0.05),
        })
    
    if "faster" in prompt:
        param_updates.update({
            "points_per_side": max(16, DEFAULT_PARAMS["points_per_side"] // 2),
            "iou_threshold": max(0.5, DEFAULT_PARAMS["iou_threshold"] - 0.1),
            "stability_threshold": max(0.5, DEFAULT_PARAMS["stability_threshold"] - 0.1),
        })
    
    return param_updates

async def process_image_with_sam(
    image_data: bytes,
    mime_type: str = "image/png",
    prompt: str = "",
    **sam_params
) -> Tuple[Optional[bytes], str]:
    """
    Process an image with SAM using the provided parameters.
    
    Args:
        image_data: Raw image bytes
        mime_type: MIME type of the image
        prompt: Natural language prompt for parameter adjustment
        **sam_params: Additional SAM parameters to override defaults
        
    Returns:
        Tuple of (segmented_image_bytes, analysis_text)
    """
    # Get default parameters
    params = DEFAULT_PARAMS.copy()
    
    # Parse prompt for parameter updates
    param_updates = parse_prompt_for_parameters(prompt)
    params.update(param_updates)
    
    # Update with any explicitly provided parameters
    params.update(sam_params)
    
    # Validate parameters
    params["points_per_side"] = max(1, min(128, int(params["points_per_side"])))
    params["iou_threshold"] = max(0.1, min(1.0, float(params["iou_threshold"])))
    params["stability_threshold"] = max(0.1, min(1.0, float(params["stability_threshold"])))
    params["box_nms_threshold"] = max(0.1, min(1.0, float(params["box_nms_threshold"])))
    
    try:
        # Save image to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
            tmp.write(image_data)
            tmp_path = tmp.name
        
        try:
            # Initialize Gradio client
            client = Client("https://evitsam.hanlab.ai/")
            
            # Call the SAM model with the image and parameters
            result = client.predict(
                param_0=handle_file(tmp_path),
                param_2=params["points_per_side"],
                param_3=params["iou_threshold"],
                param_4=params["stability_threshold"],
                param_5=params["box_nms_threshold"],
                api_name="/lambda_3"
            )
            
            # Read the result image
            if os.path.exists(result):
                with open(result, "rb") as f:
                    result_image = f.read()
                
                # Generate analysis text
                analysis = (
                    f"Processed image with SAM parameters:\n"
                    f"- Points per side: {params['points_per_side']}\n"
                    f"- IoU threshold: {params['iou_threshold']:.2f}\n"
                    f"- Stability threshold: {params['stability_threshold']:.2f}\n"
                    f"- Box NMS threshold: {params['box_nms_threshold']:.2f}"
                )
                
                return result_image, analysis
            else:
                return None, "Failed to process image: No output file was generated"
                
        finally:
            # Clean up temporary files
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            if os.path.exists(result):
                os.unlink(result)
                
    except Exception as e:
        return None, f"Error processing image with SAM: {str(e)}"

async def get_image(
    content: list[dict[str, Any]], 
    tool: dict[str, Any] | None = None
) -> Tuple[Optional[bytes], str]:
    """
    Extract image from content and process it with SAM.
    
    Args:
        content: List of content items that may contain images and text
        tool: Optional tool parameters
        
    Returns:
        Tuple of (segmented_image_bytes, analysis_text)
    """
    prompt = ""
    image_data = None
    mime_type = "image/png"
    
    # Extract text prompt and image data
    for item in content:
        if item.get("type") == "text":
            prompt = item.get("text", "")
            print(f"Extracted prompt: {prompt}")  # Debug log
        elif item.get("type") == "resource" and item.get("mime_type", "").startswith("image/"):
            image_data = base64.b64decode(item["contents"])
            mime_type = item.get("mime_type", "image/png")
    
    if image_data:
        return await process_image_with_sam(image_data, mime_type, prompt=prompt, **(tool or {}))
    
    return None, "No valid image found in the content"