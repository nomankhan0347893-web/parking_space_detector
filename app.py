# File: app.py
# Parking Space Detector - Fixed False Detection Version
# Author: Noman Khan

import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import io
import base64
import os

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Parking Space Detector",
    page_icon="🅿️",
    layout="wide"
)

MODEL_PATH    = "models/best.pt"
PROFILE_IMAGE = "profile.jpg"   # put your photo here (or leave missing — handled safely)


# ─── LOAD MODEL ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


# ─── DETECTION — WITH FALSE-POSITIVE FILTERS ──────────────────────────────────
def run_detection(
    image_bgr  : np.ndarray,
    model,
    conf       : float,
    min_area_pct: float,   # % of image area — kills tiny ghost boxes
    max_area_pct: float,   # % of image area — kills road-car huge boxes
    min_ar     : float,    # min width/height ratio
    max_ar     : float,    # max width/height ratio
    imgsz      : int = 640
):
    """
    WHY EACH FILTER EXISTS
    ──────────────────────────────────────────────────────────────────────
    Problem 1 → Tiny green boxes on random image patches
        Fix    → min_area_pct: box must be at least X% of image area
                 (a real parking space is never a 10x10 pixel box)

    Problem 2 → Car on road detected as occupied
        Fix    → max_area_pct: box cannot be more than X% of image
                 (a road car fills a huge area, parking spots don't)
               → max_ar: very wide boxes = road-side car, not a space
               → raise confidence slider

    Problem 3 → Real spaces missed
        Fix    → lower confidence slider in sidebar
                 (model is being too strict — 0.35–0.45 works better)
    ──────────────────────────────────────────────────────────────────────
    """

    img_h, img_w = image_bgr.shape[:2]
    img_area     = img_h * img_w

    # Resize for inference if needed
    if max(img_h, img_w) > imgsz:
        scale     = imgsz / max(img_h, img_w)
        new_w     = int(img_w * scale)
        new_h     = int(img_h * scale)
        src       = cv2.resize(image_bgr, (new_w, new_h))
    else:
        src, new_w, new_h = image_bgr, img_w, img_h

    results   = model.predict(source=src, conf=conf, imgsz=imgsz, verbose=False)
    annotated = image_bgr.copy()

    free_count = 0
    occ_count  = 0
    filtered   = 0

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            cls_id     = int(box.cls)
            confidence = float(box.conf)

            if cls_id not in (0, 1):
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Scale coords back to original image size
            if max(img_h, img_w) > imgsz:
                x1 = int(x1 * img_w / new_w)
                y1 = int(y1 * img_h / new_h)
                x2 = int(x2 * img_w / new_w)
                y2 = int(y2 * img_h / new_h)

            box_w    = max(x2 - x1, 1)
            box_h    = max(y2 - y1, 1)
            box_area = box_w * box_h
            ar       = box_w / box_h

            # ── FILTER 1: too small → ghost box / road marking noise ──────────
            if (box_area / img_area) * 100 < min_area_pct:
                filtered += 1
                continue

            # ── FILTER 2: too large → car on road, not a parking space ────────
            if (box_area / img_area) * 100 > max_area_pct:
                filtered += 1
                continue

            # ── FILTER 3: wrong shape → not a parking space rectangle ─────────
            if ar < min_ar or ar > max_ar:
                filtered += 1
                continue

            # ── Passed all filters — draw ─────────────────────────────────────
            if cls_id == 0:
                label = "Free"
                color = (40, 200, 100)   # green
                free_count += 1
            else:
                label = "Occupied"
                color = (50, 60, 230)    # red
                occ_count += 1

            # Semi-transparent fill
            overlay = annotated.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.addWeighted(overlay, 0.15, annotated, 0.85, 0, annotated)

            # Solid border
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Label pill
            text = f"{label} {confidence:.0%}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw + 8, y1), color, -1)
            cv2.putText(annotated, text, (x1 + 4, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    total = free_count + occ_count
    return annotated, free_count, occ_count, total, filtered


# ─── PROFILE IMAGE HELPER ──────────────────────────────────────────────────────
def get_base64_image(path: str) -> str | None:
    """Returns base64 string of image, or None if file doesn't exist."""
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
def build_sidebar():
    st.sidebar.markdown("""
    <div style='text-align:center; padding:12px 0 4px 0;'>
        <span style='font-size:1.8em;'>🅿️</span><br>
        <b style='font-size:1.05em;'>Parking Detector</b><br>
        <span style='font-size:0.75em; color:gray;'>YOLOv8 · OpenCV · Streamlit</span>
    </div>
    """, unsafe_allow_html=True)
    st.sidebar.markdown("---")

    # ── Detection settings ─────────────────────────────────────────────────────
    st.sidebar.subheader("⚙️ Detection Settings")

    conf = st.sidebar.slider(
        "Confidence Threshold", 0.10, 1.00, 0.40, 0.05,
        help="Lower = detect more spaces (may add noise). Higher = stricter."
    )

    st.sidebar.markdown("---")

    # ── False-positive filters ─────────────────────────────────────────────────
    st.sidebar.subheader("🔧 False-Positive Filters")

    st.sidebar.caption(
        "These 3 filters remove wrong detections. "
        "Adjust them if you see ghost boxes or road cars."
    )

    min_area = st.sidebar.slider(
        "Min Box Size  (% of image)",
        0.1, 5.0, 0.8, 0.1,
        help="Removes tiny boxes. A parking space is never smaller than this."
    )

    max_area = st.sidebar.slider(
        "Max Box Size  (% of image)",
        5.0, 60.0, 20.0, 1.0,
        help="Removes huge boxes. Road cars cover more area than a parking space."
    )

    ar_range = st.sidebar.slider(
        "Aspect Ratio  (width ÷ height)",
        0.1, 8.0, (0.5, 4.0), 0.1,
        help="Parking spaces have a recognisable shape — not too thin, not too square."
    )
    min_ar, max_ar = ar_range

    st.sidebar.markdown("---")

    # ── Quick fix guide ────────────────────────────────────────────────────────
    st.sidebar.subheader("🛠️ Quick Fix Guide")
    st.sidebar.markdown("""
| Problem | Fix |
|---|---|
| Tiny ghost boxes | ↑ Min Box Size |
| Car on road detected | ↑ Confidence · ↓ Max Box Size |
| Real spaces missed | ↓ Confidence · ↑ Max Box Size |
| Duplicate boxes | ↓ Confidence slightly |
""")

    st.sidebar.markdown("---")

    # ── How to use ────────────────────────────────────────────────────────────
    st.sidebar.subheader("📖 How to Use")
    st.sidebar.markdown("""
1. Upload a parking lot image
2. Check the detection result
3. If wrong boxes appear → adjust filters above
4. Download the annotated result
""")

    st.sidebar.markdown("---")
    st.sidebar.caption("Noman Khan © 2025")

    return conf, min_area, max_area, min_ar, max_ar


# ─── FOOTER ───────────────────────────────────────────────────────────────────
def render_footer():
    img_b64  = get_base64_image(PROFILE_IMAGE)
    img_html = (
        f'<img class="profile-img" src="data:image/jpeg;base64,{img_b64}">'
        if img_b64
        else '<div class="profile-avatar">NK</div>'   # fallback initials
    )

    st.markdown("---")
    st.markdown(f"""
    <style>
        .profile-card   {{ text-align:center; padding:30px 20px 10px 20px; }}
        .profile-img    {{ width:110px; height:110px; border-radius:50%;
                           object-fit:cover; border:3px solid #4CAF50; margin-bottom:12px; }}
        .profile-avatar {{ width:110px; height:110px; border-radius:50%;
                           background:linear-gradient(135deg,#1d4ed8,#059669);
                           color:white; font-size:2em; font-weight:800;
                           display:flex; align-items:center; justify-content:center;
                           margin:0 auto 12px auto; border:3px solid #4CAF50; }}
        .footer-links a {{ text-decoration:none; margin:0 10px;
                           color:#4CAF50; font-weight:600; font-size:0.9em; }}
        .footer-links a:hover {{ text-decoration:underline; }}
        .footer-copy    {{ color:gray; font-size:0.78em; margin-top:14px; }}
    </style>

    <div class="profile-card">
        {img_html}
        <h3 style="margin:4px 0;">Noman Khan</h3>
        <p style="color:gray; margin:2px 0; font-size:0.9em;">
            AI / ML Developer · Computer Vision · YOLOv8
        </p>
        <div class="footer-links" style="margin-top:12px;">
            <a href="https://github.com/YOUR_USERNAME" target="_blank">
                <!-- GitHub -->
                <svg style="vertical-align:middle;" height="16" viewBox="0 0 16 16"
                     width="16" fill="currentColor">
                  <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59
                           .4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37
                           -2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15
                           -.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87
                           .87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89
                           -3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08
                           -2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0
                           1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16
                           1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75
                           -3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01
                           2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42
                           -3.58-8-8-8z"/>
                </svg>
                GitHub
            </a>
            <a href="https://linkedin.com/in/YOUR_PROFILE" target="_blank">
                <!-- LinkedIn -->
                <svg style="vertical-align:middle;" height="16" viewBox="0 0 24 24"
                     width="16" fill="currentColor">
                  <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037
                           -1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667
                           H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37
                           -1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337
                           7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92
                           -2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0
                           1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9
                           h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729
                           v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24
                           23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/>
                </svg>
                LinkedIn
            </a>
            <a href="mailto:your@email.com">📧 Contact</a>
        </div>
        <p class="footer-copy">Built with ❤️ using YOLOv8 + OpenCV + Streamlit</p>
    </div>
    """, unsafe_allow_html=True)


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    # Sidebar returns all filter values
    conf, min_area, max_area, min_ar, max_ar = build_sidebar()

    # Header
    st.title("🅿️ Parking Space Occupancy Detector")
    st.caption("AI-powered smart parking · YOLOv8 · Real-time detection")
    st.markdown("---")

    # Load model
    try:
        model = load_model()
    except Exception as e:
        st.error(f"❌ Cannot load model: {e}")
        st.info("Make sure `models/best.pt` exists. Run `train.py` first.")
        render_footer()
        return

    # Upload
    uploaded = st.file_uploader(
        "Upload a parking lot image",
        type=["jpg", "jpeg", "png"]
    )

    # Empty state — show footer even without image
    if uploaded is None:
        st.info("⬆️ Upload a parking image to start detection.")
        render_footer()
        return

    # Decode image
    raw      = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img_bgr  = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Run detection with all filters
    with st.spinner("🔍 Detecting parking spaces…"):
        annotated_bgr, free_count, occ_count, total, filtered = run_detection(
            img_bgr, model, conf,
            min_area, max_area, min_ar, max_ar
        )
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    # ── Metrics ───────────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    c1.metric("🅿️  Total Spots",    total)
    c2.metric("🟢  Free Spots",     free_count,
              delta=f"{free_count/total*100:.0f}% available" if total else None,
              delta_color="normal")
    c3.metric("🔴  Occupied Spots", occ_count,
              delta=f"{occ_count/total*100:.0f}% in use" if total else None,
              delta_color="inverse")

    # Occupancy bar
    if total > 0:
        fp = free_count / total * 100
        op = occ_count  / total * 100
        st.markdown(f"""
        <div style="margin:10px 0 4px 0;">
            <div style="display:flex; justify-content:space-between;
                        font-size:0.85em; margin-bottom:5px;">
                <span style="color:#28C864;font-weight:600;">🟢 Free {fp:.1f}%</span>
                <span style="color:#E03C32;font-weight:600;">🔴 Occupied {op:.1f}%</span>
            </div>
            <div style="background:#e5e7eb; border-radius:8px;
                        height:18px; overflow:hidden; display:flex;">
                <div style="width:{fp:.1f}%; background:linear-gradient(90deg,#059669,#34d399);"></div>
                <div style="width:{op:.1f}%; background:linear-gradient(90deg,#dc2626,#f87171);"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Filtered notice
    if filtered > 0:
        st.caption(
            f"🔧 {filtered} detection(s) removed by filters "
            f"(ghost boxes / road cars). Adjust sidebar sliders if needed."
        )

    # Status message
    if total > 0:
        op = occ_count / total * 100
        if   op >= 90: st.error("🚨 Parking lot almost FULL!")
        elif op >= 70: st.warning("⚠️ Getting busy — limited spaces remain.")
        elif op >= 40: st.info("ℹ️ Moderately occupied — some spaces available.")
        else:          st.success("✅ Plenty of spaces available!")
    else:
        st.warning(
            "⚠️ No spaces detected. "
            "Try: lower Confidence · raise Max Box Size · check model is trained."
        )

    st.divider()

    # ── Images ────────────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.image(img_rgb,       caption="📷 Original",  use_container_width=True)
    with col2:
        st.image(annotated_rgb, caption="🔍 Detected", use_container_width=True)

    # ── Download ──────────────────────────────────────────────────────────────
    buf = io.BytesIO()
    Image.fromarray(annotated_rgb).save(buf, format="PNG")
    st.download_button(
        "📥 Download Result",
        data      = buf.getvalue(),
        file_name = "parking_result.png",
        mime      = "image/png",
        use_container_width=True
    )

    render_footer()


if __name__ == "__main__":
    main()