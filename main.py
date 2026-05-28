import io
import cv2
import requests
import numpy as np
import gradio as gr
from PIL import Image
import threading
import time

# -------------------------------------------------------------------------
# GLOBAL PERFORMANCE, SCENE, & TEMPORAL CACHE
# -------------------------------------------------------------------------
cached_texture_pattern = None
cached_final_composited = None
cached_simulated_frame = None
cached_confusion_mask = None

prev_low_res_hist = None
accumulated_mask = None
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
HF_API_TOKEN = "your_hf_token_here"
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
        return 0.0
    similarity = cv2.compareHist(prev_low_res_hist, hist, cv2.HISTCMP_CORREL)
    prev_low_res_hist = hist
    return similarity


# -------------------------------------------------------------------------
# UNIVERSAL PIPELINE CONTROLLER ENGINE (STATIC + STREAMING)
# -------------------------------------------------------------------------
def process_universal_pipeline(frame, texture_dropdown):
    global cached_texture_pattern, cached_final_composited, cached_simulated_frame, cached_confusion_mask, accumulated_mask, last_api_call_time

    # 1. التحقق الآمن من جودة ومصدر الإطار (Safe Fallback Validation)
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        try:
            # محاولة قراءة صورة الفلتر الافتراضية test.jpg الموجودة في المشروع المكتبي
            frame = cv2.imread("test.jpg")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            status_text = "🟢 Mode: Default Static Test Image Active"
        except:
            # بناء بيئة اختبار هندسية ديناميكية لمنع شاشات الانهيار البيضاء
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.circle(frame, (320, 240), 120, (200, 50, 50), -1)
            cv2.rectangle(frame, (200, 350), (440, 400), (50, 180, 50), -1)
            status_text = "⚠️ Mode: Internal Geometry Matrix Fallback Active (test.jpg missing)"
    else:
        status_text = "⚡ Mode: Live Active Webcam Stream Engaged"
        # تصحيح تدرج الألوان المتدفق حياً
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    h, w, _ = frame.shape
    low_res = cv2.resize(frame, (320, 240))

    gray_low_res = cv2.cvtColor(low_res, cv2.COLOR_RGB2GRAY)
    mean_brightness, std_variance = cv2.meanStdDev(gray_low_res)
    mean_brightness = float(mean_brightness[0][0])
    std_variance = float(std_variance[0][0])

    adaptive_threshold = int(np.clip(35 - (mean_brightness * 0.1), 12, 45))
    adaptive_alpha = float(np.clip(0.20 + (std_variance * 0.008), 0.30, 0.75))
    adaptive_scene_limit = float(np.clip(0.97 + (mean_brightness * 0.0001), 0.96, 0.99))

    scene_similarity = calculate_scene_similarity(low_res)
    scene_has_changed = scene_similarity < adaptive_scene_limit

    if not scene_has_changed and cached_final_composited is not None:
        return frame, cached_simulated_frame, cached_confusion_mask, cached_final_composited, f"🟢 Cached State | {status_text}"

    low_res_sim = simulate_deuteranopia(low_res)
    orig_lab = cv2.cvtColor(low_res, cv2.COLOR_RGB2LAB)
    sim_lab = cv2.cvtColor(low_res_sim.astype(np.uint8), cv2.COLOR_RGB2LAB)
    diff = cv2.addWeighted(cv2.absdiff(orig_lab[:, :, 1], sim_lab[:, :, 1]), 0.5,
                           cv2.absdiff(orig_lab[:, :, 2], sim_lab[:, :, 2]), 0.5, 0)
    _, low_res_mask = cv2.threshold(diff, adaptive_threshold, 255, cv2.THRESH_BINARY)

    num_labels, labels_im, stats, _ = cv2.connectedComponentsWithStats(low_res_mask)
    filtered_low_res_mask = np.zeros_like(low_res_mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > 150:
            filtered_low_res_mask[labels_im == i] = 255

    if accumulated_mask is None or accumulated_mask.shape != filtered_low_res_mask.shape:
        accumulated_mask = filtered_low_res_mask.astype(np.float32)
    else:
        cv2.accumulateWeighted(filtered_low_res_mask, accumulated_mask, 0.15)

    stabilized_low_res_mask = cv2.threshold(accumulated_mask.astype(np.uint8), 127, 255, cv2.THRESH_BINARY)[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    smoothed_low_res_mask = cv2.morphologyEx(stabilized_low_res_mask, cv2.MORPH_CLOSE, kernel)
    smoothed_low_res_mask = cv2.bilateralFilter(smoothed_low_res_mask, d=5, sigmaColor=75, sigmaSpace=75)

    current_time = time.time()
    if (current_time - last_api_call_time > 3.0) and not api_lock.locked():
        last_api_call_time = current_time
        threading.Thread(
            target=fetch_ai_texture_async,
            args=(low_res, smoothed_low_res_mask, texture_dropdown),
            daemon=True
        ).start()

    final_mask = cv2.resize(smoothed_low_res_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    cached_confusion_mask = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2RGB)
    cached_simulated_frame = cv2.resize(low_res_sim, (w, h)).astype(np.uint8)

    if cached_texture_pattern is not None:
        local_texture = cv2.resize(cached_texture_pattern, (w, h))
    else:
        local_texture = frame

    region_mean_intensity = cv2.mean(gray_low_res, mask=smoothed_low_res_mask)[0]
    if region_mean_intensity < 127:
        local_texture = cv2.bitwise_not(local_texture)

    cached_final_composited = apply_alpha_blending(
        original_rgb=frame,
        ai_textured_rgb=local_texture,
        binary_mask_rgb=cached_confusion_mask,
        alpha=adaptive_alpha
    )

    full_status_report = (
        f"{status_text}\n"
        f"📊 Environment Profiles: Brightness={mean_brightness:.1f} | Variance={std_variance:.1f}\n"
        f"⚙️ Active Configurations: Mask Threshold={adaptive_threshold} | Blending Alpha={adaptive_alpha:.2f}"
    )

    return frame, cached_simulated_frame, cached_confusion_mask, cached_final_composited, full_status_report


# -------------------------------------------------------------------------
# MODULE 1: UI Layout Dashboard (واجهة معززة هجينة تلقائية)
# -------------------------------------------------------------------------
# نقوم بإنشاء المخرجات أولاً ليتمكن المحرك الافتراضي من تعبئتها فور التشغيل الأول
out_orig = gr.Image(label="1. Original Target Frame", interactive=False)
out_sim = gr.Image(label="2. Perceptual CVD Simulation", interactive=False)
out_mask = gr.Image(label="3. Morphologically Closed Mask", interactive=False)
out_final = gr.Image(label="4. Safe-Composited Accessibility Output", interactive=False)
scene_telemetry = gr.Textbox(label="System Architecture Metrics", interactive=False, lines=4)

with gr.Blocks(title="ChromaSight AI - Universal Dashboard") as demo:
    gr.Markdown(
        """
        # 👁️ ChromaSight AI — Universal Hybrid Stream Dashboard
        ### Auto-Init Static Test Frame Pipeline with Seamless Hot-Swappable Live Webcam Execution
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🛠️ System Control Interface")

            # تم تمكين الكاميرا مع السماح لـ Gradio بالتحميل الذاتي الافتراضي
            webcam_input = gr.Image(
                sources=["webcam"],
                type="numpy",
                streaming=True,
                label="Active Camera Buffer Input"
            )
            texture_dropdown = gr.Dropdown(choices=["dots", "hatching", "voronoi"], value="dots",
                                           label="AI Texture Pattern Style")

            gr.Markdown("### 📡 System Telemetry Monitor")
            scene_telemetry.render()

        with gr.Column(scale=3):
            gr.Markdown("### 📊 Synchronized Vision Pipelines")
            with gr.Row():
                out_orig.render()
                out_sim.render()
            with gr.Row():
                out_mask.render()
                out_final.render()

    # الإجراء الأول (Auto-Initialization on Load): يتم تفعيل الأنابيب الأربعة تلقائياً بصورة ديفولت فور فتح الصفحة
    demo.load(
        fn=process_universal_pipeline,
        inputs=[gr.State(None), texture_dropdown],
        outputs=[out_orig, out_sim, out_mask, out_final, scene_telemetry]
    )

    # الإجراء الثاني (Live Streaming Switch): عندما يقرر المستخدم النقر على الكاميرا، يبدأ البث المباشر فوراً وتتحدث الواجهة
    webcam_input.stream(
        fn=process_universal_pipeline,
        inputs=[webcam_input, texture_dropdown],
        outputs=[out_orig, out_sim, out_mask, out_final, scene_telemetry],
        queue=True,
        time_limit=15
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(), share=False)