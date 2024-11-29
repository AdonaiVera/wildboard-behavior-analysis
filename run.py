from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict
import os
import cv2

# Define client-specific variables
image_entries = [{
    "input_folder": "input/image",
    "output_folder": "output/output_image",
    "confidence_threshold": 0.3,
    "model_path": "models/yolov8_thermal.pt",
    "iou_threshold": 0.7,
}]

def process_image_entries_in_parallel(image_entries: List[Dict]) -> None:
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_image_entry, entry) for entry in image_entries]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Image entry processing failed: {e}")

def process_image_entry(entry: Dict) -> None:
    from app import process_image
    
    input_folder = entry.get("input_folder")
    output_folder = entry.get("output_folder")
    confidence_threshold = entry.get("confidence_threshold")
    iou_threshold = entry.get("iou_threshold")
    model_path = entry.get("model_path")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for image_name in os.listdir(input_folder):
        if image_name.endswith((".jpg", ".jpeg", ".png")):
            input_image_path = os.path.join(input_folder, image_name)
            output_image_path = os.path.join(output_folder, image_name)
            
            process_image(
                model_path=model_path,
                input_image_path=input_image_path,
                output_image_path=output_image_path,
                confidence_threshold=confidence_threshold,
                iou_threshold=iou_threshold,
            )

def main():
    process_image_entries_in_parallel(image_entries)

if __name__ == "__main__":
    main()