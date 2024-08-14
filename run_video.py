from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict
import numpy as np

# Define client-specific variables
camera_entry = [{
    "url": "input/wild_pig_demo.mp4",
    "confidence_threshold": 0.4,
    "target_video_path": "output/result_yolov8.mp4",
    "switch_model": "models/yolov8_v3.pt", # "yolov8l-worldv2.pt"
    "iou_threshold": 0.7,
}]

def process_camera_entries_in_parallel(camera_entries: List[Dict]) -> None:
    # Use ProcessPoolExecutor to execute tasks in parallel
    with ProcessPoolExecutor() as executor:
        # Schedule the process_camera_entry function to be executed for each camera entry
        futures = [executor.submit(process_camera_entry, entry) for entry in camera_entries]

        # Wait for all the futures to complete
        for future in as_completed(futures):
            try:
                # Get the result of the future
                future.result()
            except Exception as e:
                print(f"Camera entry processing failed: {e}")


def process_camera_entry(entry: Dict) -> None:
    from app import process_video 
    
    # Extract necessary variables from the entry, add more as needed
    stream_url = entry.get("url")
    confidence_threshold = entry.get("confidence_threshold")
    iou_threshold = entry.get("iou_threshold")
    target_video_path = entry.get("target_video_path")
    model = entry.get("switch_model")

    # Call the model processing function
    process_video(
        source_weights_path=model,
        stream_url=stream_url,
        target_video_path=target_video_path,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        ignore_seconds_start=0, 
        ignore_seconds_end=-1, 
    )

def main():
    # Process the camera entries in parallel
    process_camera_entries_in_parallel(camera_entry)

if __name__ == "__main__":
    main()