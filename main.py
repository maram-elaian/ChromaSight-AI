import io
import cv2
import requests
import numpy as np
import gradio as gr
from PIL import Image
import threading
import time

# -------------------------------------------------------------------------
# GLOBAL PERFORMANCE & SCENE CACHE
# -------------------------------------------------------------------------
cached_texture_pattern = None
cached_final_composited = None
cached_simulated_frame = None
cached_confusion_mask = None

prev_low_res_hist = None
last_api_call_time = 0
api_lock = threading.Lock()

# -------------------------------------------------------------------------
# MODULE 2 & 4: Core Core Math Engines
# -------------------------------------------------------------------------
M = np.array([
    [0.430, 0.720, -0.150],
    [0.340, 0.620, 0.040],
    [-0.020, 0.030, 0.990]
], dtype=np.float32)


def simulate_deuteranopia(frame):
    img = frame.astype(np.float32) / 255.0
    out = cv2.transform(img, M)
    return np.clip(out, 0, 1) * 255


def apply_alpha_blending(original_rgb, ai_textured_rgb, binary_mask_rgb, alpha):
    img_orig = original_rgb.astype(np.float32)
    img_text = ai_textured_rgb.astype(np.float32)
    normalized_mask = (binary_mask_rgb.astype(np.float32) / 255.0) * alpha
    blended_output = (1.0 - normalized_mask) * img_orig + normalized_mask * img_text
    return np.clip(blended_output, 0, 255).astype(np.uint8)


# -------------------------------------------------------------------------
# MODULE 3: Asynchronous Cloud Requester
# -------------------------------------------------------------------------
HF_API_TOKEN = "your_hf_token_here"  # ضعي التوكن الخاص بكِ هنا
API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}


def fetch_ai_texture_async(low_res_frame, low_res_mask, texture_style):
    global cached_texture_pattern
    prompt_mapping = {
        "dots": "Clean minimalist monochrome micro-dot matrix pattern, uniform spacing, high contrast, fine vector lines",
        "hatching": "Highly detailed fine monochrome cross-hatching geometric line pattern, crisp clean thin strokes",
        "voronoi": "Organic Voronoi diagram cell pattern, ultra-fine monochrome continuous lines"
    }
    try:
        _, img_encoded = cv2.imencode('.png', low_res_frame)
        _, mask_encoded = cv2.imencode('.png', low_res_mask)
        payload = {
            "inputs": {
                "image": img_encoded.tobytes(),
                "mask_image": mask_encoded.tobytes(),
                "prompt": prompt_mapping.get(texture_style, "dots"),
                "negative_prompt": "blurry, colorful, ugly, chaotic, gradients, photorealistic background",
                "num_inference_steps": 4,
                "guidance_scale": 1.0
            }
        }
        response = requests.post(API_URL, headers=headers, json=payload, timeout=5)
        if response.status_code == 200:
            generated_image = Image.open(io.BytesIO(response.content)).convert("RGB")
            with api_lock:
                cached_texture_pattern = np.array(generated_image)
    except Exception as e:
        print(f"Async Cloud Error: {e}")


# -------------------------------------------------------------------------
# LIGHTWEIGHT SCENE CHANGE DETECT ENGINE
# -------------------------------------------------------------------------
def calculate_scene_similarity(current_low_res):
    global prev_low_res_hist
    hist = cv2.calcHist([current_low_res], [0], None, [64], [0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    if prev_low_res_hist is None:
        prev_low_res_hist = hist
        return 0.0  # مشهد جديد بالكامل

    similarity = cv2.compareHist(prev_low_res_hist, hist, cv2.HISTCMP_CORREL)
    prev_low_res_hist = hist
    return similarity


# -------------------------------------------------------------------------
# REAL-TIME ADAPTIVE PIPELINE CONTROLLER (محرك المعايرة الذاتية والتكيف)
# -------------------------------------------------------------------------
def process_live_webcam_stream(frame, texture_dropdown):
    global cached_texture_pattern, cached_final_composited, cached_simulated_frame, cached_confusion_mask, last_api_call_time

    if frame is None:
        return None, None, None, None, "Offline Telemetry"

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, _ = frame.shape

    # 1. تصغير الأبعاد السريع
    low_res = cv2.resize(frame, (320, 240))

    # 2. خطوة التحليل الإحصائي للبيئة المحيطة (Adaptive Tuning Calculation)
    # حساب متوسط السطوع والانحراف المعياري للألوان محلياً في أجزاء من الملي ثانية
    gray_low_res = cv2.cvtColor(low_res, cv2.COLOR_RGB2GRAY)
    mean_brightness, std_variance = cv2.meanStdDev(gray_low_res)
    mean_brightness = float(mean_brightness[0][0])
    std_variance = float(std_variance[0][0])

    # معادلات التكيف الذاتي (Mathematical Mapping Profiles):
    # أ) عتبة القناع تتناسب عكسياً مع الإضاءة (إضاءة خافتة تعني حساسية أعلى = threshold منخفض)
    adaptive_threshold = int(np.clip(35 - (mean_brightness * 0.1), 12, 45))

    # ب) معامل الدمج يتناسب طردياً مع تشتت الألوان لضمان بروز النمط في البيئات المعقدة
    adaptive_alpha = float(np.clip(0.20 + (std_variance * 0.008), 0.30, 0.75))

    # ج) عتبة حركة المشهد تتوازن مع السطوع لمنع نويز الظلال من تدمير الكاش
    adaptive_scene_limit = float(np.clip(0.97 + (mean_brightness * 0.0001), 0.96, 0.99))

    # 3. فحص حركة المشهد بناءً على العتبة التكيفية المستخرجة
    scene_similarity = calculate_scene_similarity(low_res)
    scene_has_changed = scene_similarity < adaptive_scene_limit

    if not scene_has_changed and cached_final_composited is not None:
        # استخدام الكاش عند ثبات الغرفة لتوفير طاقة المعالج واختصار الوقت
        status_text = (
            f"🟢 STATIC SCENE BOUNDARIES | Reusing Memory Cache\n"
            f"📊 Environment Parameters: Brightness={mean_brightness:.1f} | Variance={std_variance:.1f}\n"
            f"⚙️ Tuning Profile: Threshold={adaptive_threshold} | Blending Alpha={adaptive_alpha:.2f}"
        )
        return frame, cached_simulated_frame, cached_confusion_mask, cached_final_composited, status_text

    # 4. إذا رصد النظام تغيراً بيئياً حقيقياً، يعيد ضبط المصفوفات فوراً
    status_text = (
        f"⚡ ADAPTIVE TUNING COMPLETED | Reprocessing Pipeline Matrix\n"
        f"📊 Environment Parameters: Brightness={mean_brightness:.1f} | Variance={std_variance:.1f}\n"
        f"⚙️ Applied Profile: Threshold={adaptive_threshold} | Blending Alpha={adaptive_alpha:.2f}"
    )

    # 5. محاكاة واستخراج قناع الالتباس الفوري باستخدام العتبة الذكية المستخرجة
    low_res_sim = simulate_deuteranopia(low_res)

    orig_lab = cv2.cvtColor(low_res, cv2.COLOR_RGB2LAB)
    sim_lab = cv2.cvtColor(low_res_sim.astype(np.uint8), cv2.COLOR_RGB2LAB)
    diff = cv2.addWeighted(cv2.absdiff(orig_lab[:, :, 1], sim_lab[:, :, 1]), 0.5,
                           cv2.absdiff(orig_lab[:, :, 2], sim_lab[:, :, 2]), 0.5, 0)

    # تطبيق الـ threshold الذي تم حسابه ذاتياً
    _, low_res_mask = cv2.threshold(diff, adaptive_threshold, 255, cv2.THRESH_BINARY)

    num_labels, labels_im, stats, _ = cv2.connectedComponentsWithStats(low_res_mask)
    filtered_low_res_mask = np.zeros_like(low_res_mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > 150:
            filtered_low_res_mask[labels_im == i] = 255

    # 6. تحديث طلبات الذكاء الاصطناعي بشكل خلفي غير متزامن
    current_time = time.time()
    if (current_time - last_api_call_time > 3.0) and not api_lock.locked():
        last_api_call_time = current_time
        threading.Thread(
            target=fetch_ai_texture_async,
            args=(low_res, filtered_low_res_mask, texture_dropdown),
            daemon=True
        ).start()

    # 7. التكبير والدمج الشفاف الفوري بناءً على معامل الـ Alpha التكيفي المستخرج
    final_mask = cv2.resize(filtered_low_res_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    cached_confusion_mask = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2RGB)
    cached_simulated_frame = cv2.resize(low_res_sim, (w, h)).astype(np.uint8)

    if cached_texture_pattern is not None:
        local_texture = cv2.resize(cached_texture_pattern, (w, h))
    else:
        local_texture = frame

    # استخدام الـ alpha الذي تم حسابه ذاتياً للتكيف البصري
    cached_final_composited = apply_alpha_blending(
        original_rgb=frame,
        ai_textured_rgb=local_texture,
        binary_mask_rgb=cached_confusion_mask,
        alpha=adaptive_alpha
    )

    return frame, cached_simulated_frame, cached_confusion_mask, cached_final_composited, status_text


# -------------------------------------------------------------------------
# MODULE 1: UI Layout Dashboard (الواجهة الأوتوماتيكية الكاملة)
# -------------------------------------------------------------------------
with gr.Blocks(title="ChromaSight AI - Adaptive Framework") as demo:
    gr.Markdown(
        """
        # 👁️ ChromaSight AI — Autonomous Self-Tuning Adaptive Framework
        ### Closed-Loop Computer Vision Pipeline with Real-Time Environmental Parameter Mapping
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🛠️ Automation Controls")

            webcam_input = gr.Image(sources=["webcam"], type="numpy", streaming=True, label="Active Video Buffer")
            texture_dropdown = gr.Dropdown(choices=["dots", "hatching", "voronoi"], value="dots",
                                           label="AI Texture Pattern Style")

            gr.Markdown("### 📡 Real-Time Calibration Telemetry")
            # شاشة عرض تقارير التعديل التلقائي التي تعكس تفكير النظام التكيفي أمام الحضور
            scene_telemetry = gr.Textbox(label="Adaptive Calibration Monitor", interactive=False, lines=4)

        with gr.Column(scale=3):
            gr.Markdown("### 📊 Real-Time Pipeline Displays")

            with gr.Row():
                out_orig = gr.Image(label="1. Original Live Stream", interactive=False)
                out_sim = gr.Image(label="2. Perceptual CVD Simulation", interactive=False)

            with gr.Row():
                out_mask = gr.Image(label="3. Dynamic Autotuned Mask", interactive=False)
                out_final = gr.Image(label="4. Adaptive Composited Output (Module 4)", interactive=False)

    webcam_input.stream(
        fn=process_live_webcam_stream,
        inputs=[webcam_input, texture_dropdown],
        outputs=[out_orig, out_sim, out_mask, out_final, scene_telemetry],
        queue=True,
        time_limit=15
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(), share=False)