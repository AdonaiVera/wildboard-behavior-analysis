from typing import List, Tuple
import cv2
import supervision as sv
from ultralytics import YOLO
import numpy as np
from collections import defaultdict

def process_video(
    source_weights_path: str,
    stream_url: str,
    target_video_path: str,
    confidence_threshold: float = 0.3,
    iou_threshold: float = 0.7,
    ignore_seconds_start: int = 0,  # Number of seconds to ignore at the start
    ignore_seconds_end: int = -1,  # Number of seconds to ignore at the end
    trajectory_length: int = 30  # Number of trajectory points to show
) -> None:
    model = YOLO(source_weights_path)
 
    if source_weights_path == "yolov8l-worldv2.pt":
        model.set_classes(["wild-pig"])

    cap = cv2.VideoCapture(stream_url)
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ignore_frames_start = ignore_seconds_start * fps
    ignore_frames_end = ignore_seconds_end * fps if ignore_seconds_end != -1 else 0
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(target_video_path, fourcc, fps, (frame_width, frame_height))

    current_frame = 0
    
    # Dictionary to store trajectories
    trajectories = defaultdict(list)

    # Loop through the video frames
    while cap.isOpened():
        try:
            # Read a frame from the video
            success, frame = cap.read()
            current_frame += 1

            if success:
                # Check if current frame is within the range to be ignored
                if current_frame <= ignore_frames_start or (ignore_seconds_end != -1 and current_frame > total_frames - ignore_frames_end):
                    # Write the original frame without processing
                    out.write(frame)
                else:
                    # Run YOLOv8 tracking on the frame, persisting tracks between frames
                    results = model.track(frame, conf=confidence_threshold, persist=True)
                    
                    # Initialize variable to track the leader's ID
                    leader_id = None
                    max_distance = float('-inf')

                    # Extract bounding boxes and track IDs from the results
                    for track in results[0].boxes:
                        # Check if track contains valid data
                        if hasattr(track, 'id') and hasattr(track, 'xyxy'):
                            ids = track.id.tolist()  # Convert object IDs to list
                            boxes = track.xyxy.tolist()  # Convert bounding boxes to list

                            # Ensure the ids and boxes lists have the same length
                            if len(ids) == len(boxes):
                                # Iterate through each detected object
                                for obj_id, box in zip(ids, boxes):
                                    # Ensure the box contains four elements (x_min, y_min, x_max, y_max)
                                    if len(box) == 4:
                                        # Calculate the centroid of the bounding box
                                        centroid_x = int((box[0] + box[2]) / 2)
                                        centroid_y = int((box[1] + box[3]) / 2)

                                        # Append the centroid to the trajectory of the respective object
                                        trajectories[obj_id].append((centroid_x, centroid_y))

                                        # Determine the distance from the group's starting point (e.g., bottom-left)
                                        distance = np.sqrt(centroid_x**2 + centroid_y**2)
                                        if distance > max_distance:
                                            max_distance = distance
                                            leader_id = obj_id

                                        # Draw the current bounding box
                                        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)

                                        # Visualize only the last `trajectory_length` points of the trajectory
                                        for i in range(max(1, len(trajectories[obj_id]) - trajectory_length), len(trajectories[obj_id])):
                                            cv2.line(frame, trajectories[obj_id][i - 1], trajectories[obj_id][i], (128, 128, 0), 2)

                                        # Label the object
                                        label = f"ID {obj_id}"
                                        cv2.putText(frame, label, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                    else:
                                        print(f"Unexpected box format: {box}")
                            else:
                                print("Mismatched ids and boxes length.")
                        else:
                            print("Track is missing 'id' or 'xyxy'")
                    
                    # Highlight the leader with a different label
                    if leader_id is not None:
                        leader_position = trajectories[leader_id][-1]
                        cv2.putText(frame, "Leader", (leader_position[0], leader_position[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                            
                    # Calculate the overall bounding box that encapsulates all detections
                    if len(results[0].boxes.xyxy) > 0:
                        x_min = int(results[0].boxes.xyxy[:, 0].min())
                        y_min = int(results[0].boxes.xyxy[:, 1].min())
                        x_max = int(results[0].boxes.xyxy[:, 2].max())
                        y_max = int(results[0].boxes.xyxy[:, 3].max())

                        # Draw the overall bounding box on the frame
                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 255, 0), 2)

                        # Add a label
                        label = "Group of Wild-pigs"
                        cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


                    # Visualize the results on the frame
                    annotated_frame = results[0].plot()

                    # Write the annotated frame to the output video
                    out.write(annotated_frame)

                    # Display the annotated frame
                    cv2.imshow("YOLOv8 WildBoard Tracking with Trajectories", annotated_frame)

                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                # Break the loop if the end of the video is reached
                break
        except Exception as e:
            print("[INFO] Error iterating on the video, the error is {}".format(e))

    # Release the video capture and writer objects and close the display window
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def process_image(
    model_path: str,
    input_image_path: str,
    output_image_path: str,
    confidence_threshold: float = 0.3,
    iou_threshold: float = 0.7
) -> None:
    model = YOLO(model_path)
    if model_path == "yolov8l-worldv2.pt":
        model.set_classes(["wild-pig"])

    image = cv2.imread(input_image_path)
    results = model(image, conf=confidence_threshold, iou=iou_threshold)
    annotated_image = results[0].plot()
    cv2.imwrite(output_image_path, annotated_image)