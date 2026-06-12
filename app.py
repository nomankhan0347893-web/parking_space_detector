# File: app.py
# Parking Space Detector - Portfolio Style Version

import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import io
import base64

# ─── PAGE CONFIG ─────────────────────────
st.set_page_config(
    page_title="Parking Space Detector",
    layout="wide"
)

MODEL_PATH = "models/best.pt"
PROFILE_IMAGE = "profile.jpg"   # 👈 PUT YOUR IMAGE FILE HERE


# ─── LOAD MODEL ─────────────────────────
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


# ─── DETECTION FUNCTION ─────────────────────────
def run_detection(image_bgr: np.ndarray, model, conf: float, imgsz=640):
    # Optional: resize to a fixed size for more stable predictions
    h, w = image_bgr.shape[:2]
    if max(h, w) > imgsz:
        scale = imgsz / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        image_resized = cv2.resize(image_bgr, (new_w, new_h))
    else:
        image_resized = image_bgr

    results = model.predict(source=image_resized, conf=conf, imgsz=imgsz, verbose=False)
    annotated = image_bgr.copy()   # draw on original resolution later

    free_count = 0
    occ_count = 0

    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            cls_id = int(box.cls)
            confidence = float(box.conf)
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            box_width = x2 - x1
            box_height = y2 - y1
            aspect_ratio = box_width / box_height

# Real parking spots are usually wider than tall (e.g., ratio 1.5 to 3)
            if cls_id == 0 and (aspect_ratio < 0.4 or aspect_ratio > 4.0):
               continue   # skip weird-shaped free detections

            # Only class 0 (free) and 1 (occupied) are used
            if cls_id not in (0, 1):
                continue

            # No area filters – trust the model
            if cls_id == 0:
                label = "Free"
                color = (40, 200, 100)
                free_count += 1
            else:
                label = "Occupied"
                color = (50, 60, 230)
                occ_count += 1

            # Scale coordinates back to original image size (if you resized)
            if max(h, w) > imgsz:
                scale_x = w / new_w
                scale_y = h / new_h
                x1 = int(x1 * scale_x)
                y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x)
                y2 = int(y2 * scale_y)

            # Draw on the original‑size annotated image
            overlay = annotated.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.addWeighted(overlay, 0.15, annotated, 0.85, 0, annotated)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            text = f"{label} {confidence:.0%}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw + 8, y1), color, -1)
            cv2.putText(annotated, text, (x1 + 4, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    total = free_count + occ_count
    return annotated, free_count, occ_count, total

# ─── SIDEBAR ─────────────────────────
st.sidebar.title("⚙️ Controls")

conf = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5)

st.sidebar.markdown("---")
st.sidebar.subheader(" Guide")
st.sidebar.write("""
1. Upload image  
2. Adjust confidence  
3. View results  
4. Download output  
""")


# ─── PROFILE IMAGE FUNCTION (CIRCLE STYLE) ─────────────────────────
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# ─── FOOTER ABOUT SECTION (FIXED POSITION LOGIC) ─────────────────────────
def render_footer():
    img_base64 = get_base64_image(PROFILE_IMAGE)

    st.markdown("---")

    st.markdown(f"""
    <style>
    .profile-card {{
        text-align: center;
        padding: 25px;
        margin-top: 40px;
    }}

    .profile-img {{
        width: 120px;
        height: 120px;
        border-radius: 50%;
        object-fit: cover;
        border: 3px solid #4CAF50;
        margin-bottom: 10px;
    }}

    .footer-text {{
        text-align: center;
        font-size: 14px;
        color: gray;
    }}
    </style>

    <div class="profile-card">
        <img class="profile-img" src="data:image/png;base64,{img_base64}">
        <h3> Noman Khan</h3>
        <p>AI / ML Developer | Computer Vision | YOLOv8</p>
        <p>🔗 LinkedIn | 💻 GitHub |  Streamlit Projects</p>
        <p class="footer-text">
            Built with ❤️ using YOLOv8 + Streamlit
        </p>
    </div>
    """, unsafe_allow_html=True)
# ─── MAIN APP ─────────────────────────
def main():

    st.title(" Parking Space Occupancy Detector")
    st.caption("AI-powered YOLOv8 Smart Parking System")

    model = load_model()

    uploaded = st.file_uploader("Upload Parking Image", type=["jpg", "png", "jpeg"])

    if uploaded is None:
        st.info("Upload an image to start detection")
        return

    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    annotated, free_count, occ_count, total = run_detection(
        image_bgr, model, conf
    )
    

    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    # ─── METRICS ─────────────────────────
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Spots", total)
    c2.metric("Free Spots", free_count)
    c3.metric("Occupied Spots", occ_count)

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.image(image_rgb, caption="Original", use_container_width=True)

    with col2:
        st.image(annotated_rgb, caption="Detected", use_container_width=True)

    # ─── DOWNLOAD ─────────────────────────
    result_img = Image.fromarray(annotated_rgb)
    buf = io.BytesIO()
    result_img.save(buf, format="PNG")

    st.download_button(
        "📥 Download Result",
        data=buf.getvalue(),
        file_name="parking_result.png",
        mime="image/png"
    )
    
        # ─── FOOTER (ALWAYS LAST) ─────────────────────────
    render_footer()




if __name__ == "__main__":
    main()