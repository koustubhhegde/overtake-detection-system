# Overtake Detection System
This project implements a **real-time overtaking detection system** using computer vision. It processes rear-view video footage to detect vehicles and identify overtaking events using object detection and motion-based logic.
The system leverages **OpenCV and Roboflow API** to perform frame-by-frame analysis and ensures robust performance with retry-safe inference handling.

---

## Features  

- Real-time vehicle detection using Roboflow API  
- Overtaking detection using custom logic (line crossing)  
- Persistent tracking of vehicles across frames  
- Retry-safe API calls to handle network failures  
- Full video processing with output video generation  
- Visual annotations (bounding boxes + labels)  

---

## Technologies Used  

- **Python** – Core programming language  
- **OpenCV** – Video processing and visualization  
- **NumPy** – Numerical operations  
- **Roboflow Inference API** – Object detection model  

---

## How It Works  

1. Video is processed frame-by-frame  
2. Each frame is sent to Roboflow for object detection  
3. Bounding boxes are generated for detected vehicles  
4. A virtual red line is used as a reference  
5. If a vehicle crosses the line → marked as **OVERTAKING**  
6. Results are saved into an output video  

---

## How to Run  

1. Install dependencies:  
```bash
pip install opencv-python numpy inference-sdk
```
2. Update input/output paths in main.py:
```bash
input_video_path = "your_input_video.mp4"
output_video_path = "output.mp4"
```
3.Run the script:
```bash
python main.py
```

## Project Structure
```bash
overtake-detection/
│
├── main.py            # Main script
├── README.md          # Project documentation
