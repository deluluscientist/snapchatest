import os
import urllib.request

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, webrtc_streamer

# ---------------------------------------------------------------------------
# Model setup: YOLOv8n (ONNX) — lightweight, real-time on CPU.
# Uses readNetFromONNX rather than readNetFromCaffe: some OpenCV builds
# (notably conda-forge / some pip wheels) are compiled without protobuf
# support, which silently removes the Caffe/TensorFlow importers but keeps
# ONNX working. Auto-downloads on first run and caches locally.
# ---------------------------------------------------------------------------
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "yolov8n.onnx")
MODEL_URL = "https://raw.githubusercontent.com/yoobright/yolo-onnx/master/yolov8n.onnx"

INPUT_SIZE = 640  # YOLOv8n expects a square 640x640 input
PERSON_CLASS_ID = 0  # COCO class 0 = "person"

MAP_WIDTH = 480
MAP_HEIGHT = 480


@st.cache_resource
def load_model():
    os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    net = cv2.dnn.readNetFromONNX(MODEL_PATH)
    return net


st.set_page_config(page_title="Human + Terrain Scanner", layout="wide")
st.title("🎯 Human + Terrain Scanner")
st.caption("Real-world PUBG-style HUD: labels people and ground/terrain type from your camera.")

with st.spinner("Loading detection model (first run downloads ~23MB, then cached)..."):
    net = load_model()

st.sidebar.header("Detection Settings")
conf_threshold = st.sidebar.slider("Human detection confidence", 0.1, 0.9, 0.4, 0.05)
terrain_cell_size = st.sidebar.slider("Terrain grid cell size (px)", 20, 120, 60, 10)
show_terrain_grid = st.sidebar.checkbox("Show terrain grid lines", value=True)


def classify_terrain_cell(hsv_cell):
    """Cheap heuristic terrain classifier based on average HSV of a patch."""
    h = float(np.mean(hsv_cell[:, :, 0]))
    s = float(np.mean(hsv_cell[:, :, 1]))
    v = float(np.mean(hsv_cell[:, :, 2]))

    if s < 30 and v > 180:
        return "Snow/Bright", (255, 255, 255)
    if s < 35 and v > 90:
        return "Pavement/Rock", (140, 140, 140)
    if 35 <= h <= 90 and s > 40:
        return "Grass/Vegetation", (0, 160, 0)
    if 5 <= h < 35 and s >= 25:
        return "Dirt/Sand", (30, 105, 180)
    if 90 < h <= 135:
        if s < 60 and v > 150:
            return "Sky", (235, 206, 135)
        return "Water", (200, 120, 0)
    return "Other/Unknown", (90, 90, 90)


def annotate_terrain(frame_bgr, cell_size):
    h, w = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    overlay = frame_bgr.copy()

    for y in range(0, h, cell_size):
        for x in range(0, w, cell_size):
            cell = hsv[y:min(y + cell_size, h), x:min(x + cell_size, w)]
            if cell.size == 0:
                continue
            label, color = classify_terrain_cell(cell)
            cv2.rectangle(overlay, (x, y), (x + cell_size, y + cell_size), color, -1)

    # Blend the color-coded terrain overlay under the original frame
    blended = cv2.addWeighted(overlay, 0.35, frame_bgr, 0.65, 0)

    if show_terrain_grid:
        for y in range(0, h, cell_size):
            cv2.line(blended, (0, y), (w, y), (60, 60, 60), 1)
        for x in range(0, w, cell_size):
            cv2.line(blended, (x, 0), (x, h), (60, 60, 60), 1)

    return blended


def detect_humans(net, frame_bgr, conf_threshold):
    h, w = frame_bgr.shape[:2]

    # Letterbox-free resize straight to square input (frame is already square
    # since we resize to MAP_WIDTH x MAP_HEIGHT before this is called).
    blob = cv2.dnn.blobFromImage(
        frame_bgr, 1 / 255.0, (INPUT_SIZE, INPUT_SIZE), swapRB=True, crop=False
    )
    net.setInput(blob)
    output = net.forward()  # shape: (1, 84, 8400) -> [x,y,w,h, 80 class scores] per box

    predictions = np.squeeze(output).T  # -> (8400, 84)
    class_scores = predictions[:, 4:]
    class_ids = np.argmax(class_scores, axis=1)
    confidences = np.max(class_scores, axis=1)

    person_mask = (class_ids == PERSON_CLASS_ID) & (confidences > conf_threshold)
    boxes_xywh = predictions[person_mask, :4]
    scores = confidences[person_mask]

    if len(boxes_xywh) == 0:
        return []

    x_scale = w / INPUT_SIZE
    y_scale = h / INPUT_SIZE

    rects = []
    for (cx, cy, bw, bh) in boxes_xywh:
        x1 = (cx - bw / 2) * x_scale
        y1 = (cy - bh / 2) * y_scale
        rects.append([int(x1), int(y1), int(bw * x_scale), int(bh * y_scale)])

    indices = cv2.dnn.NMSBoxes(rects, scores.tolist(), conf_threshold, 0.45)

    results = []
    if len(indices) > 0:
        for i in np.array(indices).flatten():
            x, y, bw, bh = rects[i]
            x1, y1, x2, y2 = x, y, x + bw, y + bh
            results.append((x1, y1, x2, y2, float(scores[i])))

    return results


class ScannerProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.resize(img, (MAP_WIDTH, MAP_HEIGHT))

        # Terrain layer first (background), humans drawn on top
        annotated = annotate_terrain(img, terrain_cell_size)

        boxes = detect_humans(net, img, conf_threshold)
        for (x1, y1, x2, y2, confidence) in boxes:
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = f"Human {confidence * 100:.0f}%"
            cv2.rectangle(annotated, (x1, max(0, y1 - 20)), (x1 + 130, y1), (0, 0, 255), -1)
            cv2.putText(
                annotated, label, (x1 + 3, max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

webrtc_streamer(
    key="human-terrain-scanner",
    video_processor_factory=ScannerProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

st.markdown("---")
st.markdown(
    "**How it works:** a lightweight MobileNet-SSD model detects and boxes people in red; "
    "the background is split into a grid and each cell is color-coded by its average HSV "
    "into a rough terrain type (grass, dirt/sand, water, sky, pavement, snow). "
    "Tune the sidebar sliders if labels look off for your lighting/environment."
)