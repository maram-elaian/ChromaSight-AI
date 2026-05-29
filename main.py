import io
import cv2
import requests
import numpy as np
import gradio as gr
from PIL import Image
import threading
import time
import os

# -------------------------------------------------------------------------
# GLOBAL PERFORMANCE, SCENE, & TEMPORAL CACHE
# -------------------------------------------------------------------------
cached_texture_pattern = None
cached_final_composited = None
cached_simulated_frame = None
cached_confusion_mask = None

ema_low_res_mask = None
accumulated_mask = None
last_api_call_time = 0
api_lock = threading.Lock()

# نظام تتبع حالة الإدخال لضمان موازنة الحسابات البرمجية
system_mode_tracker = "STATIC"

# -------------------------------------------------------------------------
# MODULE 2 & 4: Core Math Engines (True Deuteranopia Simulation Matrix)
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
    # تعزيز دمج القناع لضمان ظهور الأنماط بشكل واضح وجذاب للمستخدم
    normalized_mask = (binary_mask_rgb.astype(np.float32) / 255.0) * alpha
    blended_output = (1.0 - normalized_mask) * img_orig + normalized_mask * img_text
    return np.clip(blended_output, 0, 255).astype(np.uint8)


# -------------------------------------------------------------------------
# ADVANCED ADDON 1: Psychophysical Color-Contrast Booster
# -------------------------------------------------------------------------
def boost_color_contrast_lab(frame, mask, chroma_boost=1.35, luma_boost=1.08):
    lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB).astype(np.float32)

    l, a, b = cv2.split(lab)

    if mask.ndim == 3:
        mask_bool = mask[:, :, 0] > 0
    else:
        mask_bool = mask > 0

    l[mask_bool] = np.clip(l[mask_bool] * luma_boost, 0, 255)

    a_shifted = a[mask_bool] - 128.0
    b_shifted = b[mask_bool] - 128.0

    a_shifted *= chroma_boost
    b_shifted *= chroma_boost

    a[mask_bool] = np.clip(a_shifted + 128.0, 0, 255)
    b[mask_bool] = np.clip(b_shifted + 128.0, 0, 255)

    boosted_lab = cv2.merge([l, a, b])

    return cv2.cvtColor(boosted_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)


# -------------------------------------------------------------------------
# LOCAL MATHEMATICAL TEXTURE ENGINE WITH TEMPORAL WAVE PULSE
# -------------------------------------------------------------------------
def generate_local_clean_texture(style, h, w, scale_factor, angle_deg, is_live):
    base_spacing = int(np.clip(18 * scale_factor, 10, 32))
    texture = np.full((h, w, 3), 255, dtype=np.uint8)

    if is_live:
        t = time.time()
        pulse_shift = int(np.sin(t * 4.0) * 8)
        dot_oscillation = np.cos(t * 5.0) * 2.0
    else:
        pulse_shift = 0
        dot_oscillation = 0

    if style == "dots":
        for y in range(0, h, base_spacing):
            for x in range(0, w, base_spacing):
                dynamic_radius = int(np.clip(5 * scale_factor + dot_oscillation, 3, 10))
                cv2.circle(texture, (x, y), dynamic_radius, (0, 0, 0), -1)

    elif style == "hatching":
        for i in range(-max(h, w), max(h, w), base_spacing):
            cv2.line(texture, (i + pulse_shift, 0), (i + h + pulse_shift, h), (0, 0, 0),
                     int(np.clip(2.5 * scale_factor, 1, 5)))

    else:
        for y in range(0, h, base_spacing):
            cv2.line(texture, (0, y + pulse_shift), (w, y + pulse_shift), (0, 0, 0), 1)
        for x in range(0, w, base_spacing):
            cv2.line(texture, (x + pulse_shift, 0), (x + pulse_shift, h), (0, 0, 0), 1)

    rot_matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    transformed_texture = cv2.warpAffine(texture, rot_matrix, (w, h), flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_REFLECT_101)
    return transformed_texture


def apply_adaptive_clahe(frame):
    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)

    lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    brightness_std = np.std(l)

    adaptive_clip = float(np.clip(2.0 + (brightness_std / 45.0), 2.0, 5.0))
    h, w = l.shape
    tile_size = max(8, int(min(h, w) / 32))

    clahe = cv2.createCLAHE(clipLimit=adaptive_clip, tileGridSize=(tile_size, tile_size))
    enhanced_l = clahe.apply(l)

    merged = cv2.merge([enhanced_l, a, b])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)


# -------------------------------------------------------------------------
# UNIVERSAL PIPELINE CONTROLLER
# -------------------------------------------------------------------------
def hysteresis_threshold(diff, frame_gray, prev_mask=None):
    diff_mean = float(np.mean(diff))
    diff_std = float(np.std(diff))

    T_high = np.clip(diff_mean + (1.8 * diff_std), 15, 90)
    T_low = np.clip(diff_mean + (0.7 * diff_std), 8, 50)

    strong = (diff > T_high).astype(np.uint8)
    weak = (diff > T_low).astype(np.uint8)

    if prev_mask is None:
        prev_mask = np.zeros_like(diff, dtype=np.uint8)

    result = strong.copy()
    num_labels, labels = cv2.connectedComponents(weak)

    for i in range(1, num_labels):
        region = (labels == i)
        if np.any(strong[region]):
            result[region] = 1

    result = cv2.addWeighted(result.astype(np.float32), 0.7, prev_mask.astype(np.float32), 0.3, 0)
    return (result > 0.5).astype(np.uint8) * 255


prev_gray_frame = None
prev_warped_mask = None


def warp_mask_with_flow(mask, flow):
    h, w = mask.shape
    flow_x = flow[..., 0]
    flow_y = flow[..., 1]
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))

    map_x = (grid_x - flow_x).astype(np.float32)
    map_y = (grid_y - flow_y).astype(np.float32)

    return cv2.remap(mask.astype(np.float32), map_x, map_y, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT)


def process_universal_pipeline(frame, texture_dropdown):
    global cached_texture_pattern, cached_final_composited, cached_simulated_frame, cached_confusion_mask, accumulated_mask, system_mode_tracker
    global prev_low_res_hist, ema_low_res_mask, prev_gray_frame, prev_warped_mask

    # التعديل رقم 3: إصلاح سبب الثبات على صورة افتراضية
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        is_live_stream = False
        system_mode_tracker = "STATIC"
        status_text = "🟢 Mode: Static Image Default Active (True Flower Reference)"

        loaded = False
        for target_name in ["test.jpg", "image.png", "image.jpg", "image.jpeg"]:
            if os.path.exists(target_name):
                frame = cv2.imread(target_name)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                loaded = True
                break

        if not loaded:
            frame = np.zeros((500, 750, 3), dtype=np.uint8)
            cv2.rectangle(frame, (0, 0), (250, 500), (230, 50, 50), -1)
            cv2.rectangle(frame, (250, 0), (500, 500), (50, 220, 50), -1)
            cv2.rectangle(frame, (500, 0), (750, 500), (50, 50, 240), -1)
    else:
        status_text = "⚡ Mode: Live Stream Active [Spatial Confusion Contrast Boosted]"
        is_live_stream = True
        frame = frame.copy()

        # التعديل رقم 4: تفريغ الذاكرة الزمنية عند الانتقال للبث المباشر لمنع التجميد
        if system_mode_tracker == "STATIC":
            accumulated_mask = None
            prev_gray_frame = None
            prev_warped_mask = None
            ema_low_res_mask = None
            system_mode_tracker = "LIVE"

    h, w, _ = frame.shape
    low_res_w, low_res_h = 320, 240
    low_res = cv2.resize(frame, (low_res_w, low_res_h))

    low_res = apply_adaptive_clahe(low_res)
    gray_low_res = cv2.cvtColor(low_res, cv2.COLOR_RGB2GRAY)

    if prev_gray_frame is None:
        flow = None
    else:
        flow = cv2.calcOpticalFlowFarneback(prev_gray_frame, gray_low_res, None, 0.5, 2, 9, 3, 5, 1.2, 0)

    mean_brightness, std_variance = cv2.meanStdDev(gray_low_res)
    mean_brightness = float(mean_brightness[0][0])
    std_variance = float(std_variance[0][0])

    if is_live_stream:
        adaptive_alpha = float(np.clip(0.55 + (std_variance * 0.004), 0.60, 0.85))
    else:
        adaptive_alpha = float(np.clip(0.40 + (std_variance * 0.003), 0.45, 0.70))

    low_res_sim = simulate_deuteranopia(low_res)
    orig_lab = cv2.cvtColor(low_res, cv2.COLOR_RGB2LAB)
    sim_lab = cv2.cvtColor(low_res_sim.astype(np.uint8), cv2.COLOR_RGB2LAB)

    diff_a = cv2.absdiff(orig_lab[:, :, 1], sim_lab[:, :, 1])
    diff_b = cv2.absdiff(orig_lab[:, :, 2], sim_lab[:, :, 2])

    if prev_gray_frame is None:
        motion_weight = np.ones_like(diff_a, dtype=np.float32)
    else:
        magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        magnitude = cv2.resize(magnitude, (diff_a.shape[1], diff_a.shape[0]))
        motion_weight = cv2.normalize(magnitude, None, 0.7, 1.6, cv2.NORM_MINMAX)

    motion_weight = motion_weight.astype(np.float32)
    diff = (diff_a.astype(np.float32) + diff_b.astype(np.float32)) * 0.5
    diff = np.clip(diff * motion_weight, 0, 255).astype(np.uint8)
    diff = cv2.GaussianBlur(diff, (5, 5), 0)

    raw_mask = hysteresis_threshold(diff, gray_low_res, None)
    current_mask = raw_mask.astype(np.float32)

    if ema_low_res_mask is None or ema_low_res_mask.shape != current_mask.shape:
        ema_low_res_mask = current_mask.copy()
    else:
        alpha = np.clip(0.25 + (std_variance * 0.002), 0.2, 0.5)
        ema_low_res_mask = alpha * current_mask + (1 - alpha) * ema_low_res_mask

    # التعديل رقم 1: عتبة الـ EMA الديناميكية المعايرة
    low_res_mask = (ema_low_res_mask > 0.5 * 255).astype(np.uint8) * 255

    if flow is not None and prev_warped_mask is not None:
        warped_prev = warp_mask_with_flow(prev_warped_mask, flow)
        low_res_mask = cv2.addWeighted(low_res_mask.astype(np.float32), 0.6, np.clip(warped_prev, 0, 255), 0.4,
                                       0).astype(np.uint8)

    prev_warped_mask = low_res_mask.copy()
    prev_gray_frame = gray_low_res.copy()

    num_labels, labels_im, stats, _ = cv2.connectedComponentsWithStats(low_res_mask)
    filtered_low_res_mask = np.zeros_like(low_res_mask)
    min_area_filter = 30 if is_live_stream else 60
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > min_area_filter:
            filtered_low_res_mask[labels_im == i] = 255

    if is_live_stream:
        if accumulated_mask is None or accumulated_mask.shape != filtered_low_res_mask.shape:
            accumulated_mask = filtered_low_res_mask.astype(np.float32)
        else:
            cv2.accumulateWeighted(filtered_low_res_mask.astype(np.float32), accumulated_mask, 0.25)
    else:
        accumulated_mask = filtered_low_res_mask.astype(np.float32)

    acc_uint8 = cv2.normalize(accumulated_mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, stabilized_low_res_mask = cv2.threshold(acc_uint8, 35, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3) if is_live_stream else (5, 5))
    smoothed_low_res_mask = cv2.morphologyEx(stabilized_low_res_mask, cv2.MORPH_CLOSE, kernel)

    foveated_mask = np.zeros_like(smoothed_low_res_mask)
    center_x_low = int(low_res_w / 2)
    center_y_low = int(low_res_h / 2 + (np.sin(time.time() * 2.5) * 35 if is_live_stream else 0))
    focus_radius_low = int(min(low_res_w, low_res_h) * 0.45)
    cv2.circle(foveated_mask, (center_x_low, center_y_low), focus_radius_low, 255, -1)
    smoothed_low_res_mask = cv2.addWeighted(smoothed_low_res_mask, 0.8, foveated_mask, 0.2, 0)

    final_mask = cv2.resize(smoothed_low_res_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    cached_confusion_mask = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2RGB)

    boosted_frame = boost_color_contrast_lab(frame, final_mask)
    cached_simulated_frame = cv2.resize(low_res_sim, (w, h)).astype(np.uint8)

    gray_full = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    laplacian_var = cv2.Laplacian(gray_full, cv2.CV_64F).var()
    current_scale = float(np.clip(0.6 + (laplacian_var * 0.0008), 0.7, 1.4))

    sobelx = cv2.Sobel(gray_full, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray_full, cv2.CV_64F, 0, 1, ksize=3)
    mean_dx = cv2.mean(sobelx, mask=final_mask)[0]
    mean_dy = cv2.mean(sobely, mask=final_mask)[0]
    current_angle = float(np.degrees(np.arctan2(mean_dy, mean_dx)))

    local_texture = generate_local_clean_texture(texture_dropdown, h, w, current_scale, current_angle, is_live_stream)
    if cv2.mean(gray_full, mask=final_mask)[0] < 110:
        local_texture = cv2.bitwise_not(local_texture)

    cached_final_composited = apply_alpha_blending(boosted_frame, local_texture, cached_confusion_mask, adaptive_alpha)

    real_center_x = int(w / 2)
    real_center_y = int(h / 2 + (np.sin(time.time() * 2.5) * (h / 240 * 35) if is_live_stream else 0))
    real_focus_radius = int(min(w, h) * 0.45)
    cv2.circle(cached_final_composited, (real_center_x, real_center_y), real_focus_radius, (0, 255, 0), 2,
               lineType=cv2.LINE_AA)

    full_status_report = (
        f"{status_text}\n"
        f"🎯 Matrix Output Block: Dynamic Grid Allocation = {w}x{h}\n"
        f"🌊 Confusion Threshold Matrix: Boosted Enabled\n"
        f"📐 Spatial HUD Tracking: Dynamic Fovea Movement Enabled | Scale = {current_scale:.2f}x\n"
        f"⚙️ Sensory Feed: Ambient Lighting Coefficient = {mean_brightness:.1f}"
    )

    return frame, cached_simulated_frame, cached_confusion_mask, cached_final_composited, full_status_report


# دالة مخصصة لتهيئة وعرض الصورة الافتراضية بمجرد تشغيل السكربت
def load_initial_preview(texture_dropdown):
    return process_universal_pipeline(None, texture_dropdown)


# -------------------------------------------------------------------------
# MODULE 1: UI Layout Dashboard
# -------------------------------------------------------------------------
out_orig = gr.Image(label="1. Original Target Frame", interactive=False)
out_sim = gr.Image(label="2. Perceptual CVD Simulation", interactive=False)
out_mask = gr.Image(label="3. Morphologically Closed Mask", interactive=False)
out_final = gr.Image(label="4. High-Fidelity Accessibility Output", interactive=False)
scene_telemetry = gr.Textbox(label="Cognitive Engine Telemetry Metrics", interactive=False, lines=5)

with gr.Blocks(title="ChromaSight AI - Absolute Solution") as demo:
    gr.Markdown(
        """
        # 👁️ ChromaSight AI — High-Fidelity Foveated Cognitive Assistive Engine
        ### Resolved Pipeline Integration: Enhanced Dynamic Confusion Detection and Intelligent Foveated Layer Fusion
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🛠️ System Interface")
            webcam_input = gr.Image(sources=["webcam"], type="numpy", streaming=True,
                                    label="Active Camera Buffer Input")
            texture_dropdown = gr.Dropdown(choices=["dots", "hatching", "voronoi"], value="dots",
                                           label="AI Texture Pattern Style")

            gr.Markdown("### 📡 Cognitive Telemetry")
            scene_telemetry.render()

        with gr.Column(scale=3):
            gr.Markdown("### 📊 Synchronized Vision Pipelines")
            with gr.Row():
                out_orig.render()
                out_sim.render()
            with gr.Row():
                out_mask.render()
                out_final.render()

    # الإصلاح الجوهري: استدعاء الدالة عند تحميل الصفحة مباشرة لعرض الصور الافتراضية فوراً
    demo.load(
        fn=load_initial_preview,
        inputs=[texture_dropdown],
        outputs=[out_orig, out_sim, out_mask, out_final, scene_telemetry]
    )

    # التعديل رقم 2: معالجة بث الكاميرا الحي بشكل مستمر ومستقر بدون تجميد
    webcam_input.stream(
        fn=process_universal_pipeline,
        inputs=[webcam_input, texture_dropdown],
        outputs=[out_orig, out_sim, out_mask, out_final, scene_telemetry],
        queue=True
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(), share=False)