# ============================
#  FULL VIDEO VERSION
#  OVERTAKING DETECTION + PERSISTENCE + API RETRY
# ============================

from inference_sdk import InferenceHTTPClient
import cv2
import numpy as np
import time

# Initialize Roboflow Client
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="VtCPSbT6ne8ivWUlzU8Q"
)

# Your input/output videos
input_video_path = "SA9.MOV"
output_video_path = "DSA9.mp4"

# Red line coordinates and Y position
pt1 = (0, 900)
pt2 = (1800, 900)
red_y = 900

# Dictionary to store vehicles currently overtaking
overtaking_state = {}

# ============================
# Retry-safe inference function
# ============================

def safe_infer(image_path, retries=5, delay=1.0):
    """
    Calls Roboflow inference with retry logic.
    Prevents 502 errors from stopping the script.
    """
    for attempt in range(retries):
        try:
            return CLIENT.infer(image_path, model_id="car-night-wihfx/2")
        except Exception as e:
            print(f"Inference failed (attempt {attempt+1}/{retries}): {e}")
            time.sleep(delay)
    print("All retries failed. Skipping frame.")
    return {"predictions": []}

# ============================
# Video Processing
# ============================

cap = cv2.VideoCapture(input_video_path)

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Total frames in video: {total_frames}")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

frame_count = 0

print("Processing full video...")

while True:
    ret, frame = cap.read()
    if not ret:
        break  # video finished

    frame_count += 1
    print(f"Frame {frame_count}/{total_frames}")

    # Save raw frame for inference
    temp_path = "/content/temp_frame.jpg"
    cv2.imwrite(temp_path, frame)

    # SAFE inference call
    result = safe_infer(temp_path)

    # Draw red line AFTER inference
    cv2.line(frame, pt1, pt2, (0, 0, 255), 3)

    current_ids = []

    if "predictions" in result:
        for pred in result["predictions"]:

            x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
            class_name = pred["class"]
            conf = pred["confidence"]

            # Bounding box coordinates
            x1 = int(x - w/2)
            y1 = int(y - h/2)
            x2 = int(x + w/2)
            y2 = int(y + h/2)

            # Use center X as temp ID
            temp_id = int((x1 + x2) / 2)
            current_ids.append(temp_id)

            # OVERTAKING logic
            if y2 >= red_y:
                overtaking_state[temp_id] = True
            else:
                if temp_id in overtaking_state and y2 < red_y:
                    overtaking_state.pop(temp_id, None)

            # Draw results
            if temp_id in overtaking_state:
                color = (0, 0, 255)
                label = f"OVERTAKING {class_name} ({conf:.2f})"
            else:
                color = (0, 255, 0)
                label = f"{class_name} ({conf:.2f})"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
            )

    # Write frame
    out.write(frame)

    # Small delay to prevent server overload
    time.sleep(0.05)

cap.release()
out.release()

print("FULL VIDEO PROCESSING DONE!")
print("Saved as:", output_video_path)
