import os
import time
import asyncio
import threading
from typing import Optional

import numpy as np
import streamlit as st
from PIL import Image
import cv2
from dotenv import load_dotenv

from ultralytics import YOLO
import edge_tts
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

# .env file se GOOGLE_API_KEY load karein
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Simple fixed settings (koi sidebar config nahi)
MODEL_FILE = "yolo11m.pt"
VOICE_ID = "ur-PK-UzmaNeural"
CONFIDENCE_THRESHOLD = 0.5


# =========================================================
#                     PAGE CONFIG + STYLE
# =========================================================
st.set_page_config(
    page_title="Bina Rahnuma — Nabeena Afraad ke liye AI Guide",
    page_icon="🦯",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Nastaliq+Urdu:wght@400;700&family=Poppins:wght@400;500;600;700&display=swap');

html, body, [class*="css"]  {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background: radial-gradient(circle at top left, #1b1f3b 0%, #0d0f1c 60%, #05060c 100%);
}

/* Hero header */
.hero {
    padding: 2.2rem 2rem;
    border-radius: 20px;
    background: linear-gradient(120deg, #6C5CE7 0%, #00B4D8 100%);
    box-shadow: 0 10px 35px rgba(108, 92, 231, 0.35);
    margin-bottom: 1.8rem;
}
.hero h1 {
    color: white;
    font-size: 2.1rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
}
.hero p {
    color: #eaeaff;
    font-size: 1.02rem;
    margin: 0;
}

/* Cards */
.card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 1.4rem 1.5rem;
    backdrop-filter: blur(6px);
    margin-bottom: 1.2rem;
}
.card h3 {
    color: #ffffff;
    font-size: 1.05rem;
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* Urdu guidance box */
.guidance-box {
    background: linear-gradient(135deg, rgba(0,180,216,0.15), rgba(108,92,231,0.15));
    border: 1px solid rgba(0,180,216,0.4);
    border-radius: 18px;
    padding: 1.8rem;
    text-align: center;
    margin-bottom: 1rem;
}
.guidance-text {
    font-family: 'Noto Nastaliq Urdu', serif;
    direction: rtl;
    font-size: 2rem;
    line-height: 2.6rem;
    color: #ffffff;
    font-weight: 700;
}

/* Detection chips */
.chip {
    display: inline-block;
    padding: 0.35rem 0.9rem;
    border-radius: 999px;
    background: rgba(0, 180, 216, 0.18);
    border: 1px solid rgba(0, 180, 216, 0.45);
    color: #d7f7ff;
    font-size: 0.85rem;
    margin: 0.2rem 0.3rem 0.2rem 0;
}
.chip.warn {
    background: rgba(255, 99, 99, 0.18);
    border: 1px solid rgba(255, 99, 99, 0.5);
    color: #ffd7d7;
}

.stButton>button {
    border-radius: 12px;
    font-weight: 600;
    padding: 0.55rem 1.4rem;
    background: linear-gradient(120deg, #6C5CE7, #00B4D8);
    color: white;
    border: none;
}
.stButton>button:hover {
    opacity: 0.9;
    color: white;
}

section[data-testid="stSidebar"] {
    background: #10132a;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =========================================================
#                        SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("### 🦯 Bina Rahnuma")
    st.markdown(
        """
        <div style="color:#b8bce0; font-size:0.9rem; line-height:1.6;">
        Ye ek AI prototype hai jo image lekar
        nabeena afraad ko batata hai ke unke aage kya hai,
        aur Urdu awaaz mein guidance deta hai.
        <br><br>
        <b>Kaam ka tareeqa:</b><br>
        1. Image upload karein<br>
        2. YOLO11 objects detect karega<br>
        3. Gemini Urdu guidance banayega<br>
        4. Awaaz sunein ya download karein
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
#                    CACHED RESOURCES
# =========================================================
@st.cache_resource(show_spinner=False)
def load_yolo_model(model_name: str):
    return YOLO(model_name)


def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)


# =========================================================
#                    CORE LOGIC FUNCTIONS
# =========================================================
def get_detection_summary(results, frame_width, frame_height, model, conf_thresh):
    detections = []
    frame_area = frame_width * frame_height

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            conf = float(box.conf[0])
            if conf < conf_thresh:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x_center = (x1 + x2) / 2
            box_area = (x2 - x1) * (y2 - y1)
            area_ratio = box_area / frame_area

            if x_center < frame_width / 3:
                position = "bayen taraf"
            elif x_center > 2 * frame_width / 3:
                position = "dayen taraf"
            else:
                position = "seedhy samne"

            if area_ratio > 0.25:
                distance = "bohat qareeb"
            elif area_ratio > 0.08:
                distance = "qareeb"
            else:
                distance = "door"

            detections.append({
                "label": label,
                "position": position,
                "distance": distance,
                "confidence": round(conf, 2),
            })

    return detections


SYSTEM_PROMPT = """Aap ek madadgar assistant hain jo nabeena (blind) insaan ko chalte waqt real-time guidance dete hain.
Aapko camera se detect hone wali cheezon ki list (label, position, distance) di jayegi.

Aapka jawab:
- Sirf Urdu script mein ho (Urdu rasm-ul-khat), 1-2 chhoti lines se zyada nahi.
- Seedha aur actionable ho: batayein ke aage kya hai aur kya karna chahiye (ruk jayein, ehtiyat se chalein, bayen/dayen se nikal jayein, ya rasta saaf hai chalte rahein).
- Agar koi cheez qareeb aur seedhi samne ho to sabse pehle usay mention karein aur "ruk jayein" ya "ehtiyat" jaisa lafz zaroor use karein.
- Agar list khaali ho to sirf "rasta saaf hai, chalte rahein" ka mafhoom dein.
- Ghair zaroori tafseel na dein, seedha kaam ki baat karein.
- Poora jawab hamesha Urdu script mein likhein, Roman letters ka istemal na karein.
"""


def generate_guidance(llm, detections: list) -> str:
    if not detections:
        return "راستہ صاف ہے، چلتے رہیں۔"

    detections_sorted = sorted(
        detections,
        key=lambda d: (d["distance"] != "bohat qareeb", d["distance"] != "qareeb"),
    )
    desc_lines = [f"- {d['label']} ({d['position']}, {d['distance']})" for d in detections_sorted]
    human_prompt = "Detect hone wali cheezein:\n" + "\n".join(desc_lines)

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=human_prompt),
    ])
    return response.content.strip()


def _edge_tts_worker(text: str, filename: str, voice: str, error_box: list):
    """Alag thread + apna fresh event loop -> Streamlit/Jupyter ke chalte hue
    event loop se koi takraav nahi hota."""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            communicate = edge_tts.Communicate(text, voice)
            loop.run_until_complete(communicate.save(filename))
        finally:
            loop.close()
    except Exception as e:
        error_box.append(e)


def text_to_speech_bytes(text: str, voice: str) -> Optional[bytes]:
    filename = os.path.abspath(f"guidance_{int(time.time() * 1000)}.mp3")
    error_box = []
    t = threading.Thread(target=_edge_tts_worker, args=(text, filename, voice, error_box))
    t.start()
    t.join()

    if error_box:
        st.error(f"Awaaz banane mein masla hua: {error_box[0]}")
        return None

    if not os.path.exists(filename):
        st.error("Audio file generate nahi ho saki.")
        return None

    with open(filename, "rb") as f:
        audio_bytes = f.read()
    os.remove(filename)
    return audio_bytes


# =========================================================
#                          HEADER
# =========================================================
st.markdown(
    """
    <div class="hero">
        <h1>🦯 Bina Rahnuma — AI Chalne ka Rahnuma</h1>
        <p>Image upload karein, YOLO11 + Gemini 2.5 Flash mil kar batayenge ke aapke aage kya hai — Urdu awaaz mein.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not GOOGLE_API_KEY:
    st.warning("⚠️ `.env` file mein `GOOGLE_API_KEY` set karein taake guidance generate ho sake.")

# =========================================================
#                       IMAGE UPLOAD
# =========================================================
col_upload, col_info = st.columns([2, 1])

with col_upload:
    st.markdown('<div class="card"><h3>📷 Image Upload Karein</h3>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Image chunein (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])
    st.markdown("</div>", unsafe_allow_html=True)

with col_info:
    st.markdown(
        """
        <div class="card">
            <h3>ℹ️ Kaise kaam karta hai</h3>
            <span class="chip">1. Image upload</span><br>
            <span class="chip">2. YOLO11 detection</span><br>
            <span class="chip">3. Gemini Urdu guidance</span><br>
            <span class="chip">4. Awaaz + download</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

# =========================================================
#                    MAIN PROCESSING
# =========================================================
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    h, w = frame.shape[:2]

    with st.spinner("YOLO11 model load ho raha hai..."):
        model = load_yolo_model(MODEL_FILE)

    with st.spinner("Objects detect kiye ja rahe hain..."):
        results = model(frame, verbose=False)
        annotated = results[0].plot()
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    detections = get_detection_summary(results, w, h, model, CONFIDENCE_THRESHOLD)

    col_orig, col_annot = st.columns(2)
    with col_orig:
        st.markdown('<div class="card"><h3>🖼️ Original Image</h3>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_annot:
        st.markdown('<div class="card"><h3>🎯 Detected Objects</h3>', unsafe_allow_html=True)
        st.image(annotated_rgb, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Detection chips
    st.markdown('<div class="card"><h3>📋 Detection Summary</h3>', unsafe_allow_html=True)
    if detections:
        chips_html = ""
        for d in detections:
            css_class = "chip warn" if d["distance"] in ("bohat qareeb", "qareeb") else "chip"
            chips_html += (
                f'<span class="{css_class}">{d["label"]} • {d["position"]} • '
                f'{d["distance"]} • {int(d["confidence"]*100)}%</span>'
            )
        st.markdown(chips_html, unsafe_allow_html=True)
    else:
        st.markdown('<span class="chip">Koi object detect nahi hua ✅</span>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Guidance generation
    if GOOGLE_API_KEY:
        if st.button("🔊 Guidance Generate Karein", use_container_width=True):
            with st.spinner("Gemini se Urdu guidance banayi ja rahi hai..."):
                try:
                    llm = get_llm()
                    guidance_text = generate_guidance(llm, detections)
                except Exception as e:
                    guidance_text = None
                    st.error(f"Gemini se guidance lete waqt masla hua: {e}")

            if guidance_text:
                st.markdown(
                    f"""
                    <div class="guidance-box">
                        <div class="guidance-text">{guidance_text}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                with st.spinner("Awaaz (voice) generate ho rahi hai..."):
                    audio_bytes = text_to_speech_bytes(guidance_text, VOICE_ID)

                if audio_bytes:
                    st.audio(audio_bytes, format="audio/mp3")
                    st.download_button(
                        label="⬇️ Awaaz Download Karein (MP3)",
                        data=audio_bytes,
                        file_name="guidance.mp3",
                        mime="audio/mp3",
                        use_container_width=True,
                    )
    else:
        st.info("Guidance generate karne ke liye `.env` file mein `GOOGLE_API_KEY` set karein.")

else:
    st.markdown(
        """
        <div class="card" style="text-align:center; padding: 3rem;">
            <h3 style="justify-content:center;">👆 Upar se ek image upload karein shuru karne ke liye</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )