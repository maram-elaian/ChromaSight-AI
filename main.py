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
CACHED_TEXTURES = {}
cached_texture_pattern = None
cached_final_composited = None
cached_simulated_frame = None
cached_confusion_mask = None

ema_low_res_mask = None
accumulated_mask = None
last_api_call_time = 0
api_lock = threading.Lock()

# Global variables for tracking frames and metrics
prev_low_res_hist = None
system_mode_tracker = "STATIC"
loaded = False  # Track initial load state globally

# Real-Time FPS Tracking Metrics
fps_prev_time = 0
fps_current_rate = 0.0

# -------------------------------------------------------------------------
# CORE MATH ENGINES (True Deuteranopia Simulation Matrix)
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


def generate_local_clean_texture(style, h, w, scale_factor, angle_deg, is_live):
    base_spacing = int(np.clip(18 * scale_factor * (1 + abs(angle_deg) / 180), 8, 40))
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


def hysteresis_threshold(diff, frame_gray, prev_mask=None):
    diff_mean = float(np.mean(diff))
    diff_std = float(np.std(diff))

    T_high = np.clip(diff_mean + (1.8 * diff_std), 15, 90)
    T_low = np.clip(diff_mean + (0.7 * diff_std), 8, 50)

    strong = (diff > T_high).astype(np.uint8)
    weak = (diff > T_low).astype(np.uint8) * 255  # تحويلها لقناع ثنائي صالح لمعالجة الكونتور

    if prev_mask is None:
        prev_mask = np.zeros_like(diff, dtype=np.uint8)

    result = strong.copy()
    contours, _ = cv2.findContours(weak, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        # إنشاء قناع مؤقت وفحص مساحة الكونتور لضمان عزل النويز بمرونة عالية
        if cv2.contourArea(cnt) > 10:
            # توليد مصفوفة بكسلية للمنطقة الحالية فقط لمعاينة اتصالها بالـ strong threshold
            mask_roi = np.zeros_like(diff, dtype=np.uint8)
            cv2.drawContours(mask_roi, [cnt], -1, 1, -1)
            if np.any(strong[mask_roi > 0]):
                result[mask_roi > 0] = 1

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


def process_universal_pipeline(frame, texture_dropdown, show_foveated_circle=True, is_static_tab=False):
    global cached_texture_pattern, cached_final_composited, cached_simulated_frame, cached_confusion_mask, accumulated_mask, system_mode_tracker
    global prev_low_res_hist, ema_low_res_mask, prev_gray_frame, prev_warped_mask, loaded
    global fps_prev_time, fps_current_rate

    t_start = time.time()
    if fps_prev_time > 0:
        time_delta = t_start - fps_prev_time
        if time_delta > 0:
            fps_current_rate = 0.9 * fps_current_rate + 0.1 * (1.0 / time_delta)
    fps_prev_time = t_start

    if frame is None:
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

        status_text = "⚡ Mode: Static Media Active [Optimal Color Mapping Pipeline]"
        is_live_stream = False
    else:
        status_text = "⚡ Mode: Live Stream Active [Spatial Confusion Contrast Boosted]"
        is_live_stream = True
        frame = frame.copy()

        if system_mode_tracker == "STATIC":
            prev_gray_frame = None
            prev_warped_mask = None
            ema_low_res_mask = None
            system_mode_tracker = "LIVE"

    accumulated_mask = np.zeros((240, 320), dtype=np.float32)

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

    low_res_mask = (ema_low_res_mask > 0.5 * 255).astype(np.uint8) * 255

    if flow is not None and prev_warped_mask is not None:
        warped_prev = warp_mask_with_flow(prev_warped_mask, flow)
        low_res_mask = cv2.addWeighted(low_res_mask.astype(np.float32), 0.6, np.clip(warped_prev, 0, 255), 0.4,
                                       0).astype(np.uint8)

    prev_warped_mask = low_res_mask.copy()
    prev_gray_frame = gray_low_res.copy()


    filtered_low_res_mask = np.zeros_like(low_res_mask)
    min_area_filter = 30 if is_live_stream else 60
    contours, _ = cv2.findContours(low_res_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) > min_area_filter:
            cv2.drawContours(filtered_low_res_mask, [cnt], -1, 255, -1)

    if flow is not None:
        motion_intensity = float(np.mean(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)))
        if motion_intensity < 0.5:
            tip_text = "Static"
        elif motion_intensity < 2.5:
            tip_text = "Moving"
        else:
            tip_text = "Rapid Motion"
        motion_x = int(np.mean(flow[..., 0]) * 20)
        motion_y = int(np.mean(flow[..., 1]) * 20)
    else:
        tip_text = "Static"
        motion_x, motion_y = 0, 0

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

    total_pixels = w * h
    confusion_pixels = np.count_nonzero(final_mask)
    confusion_percentage = (confusion_pixels / total_pixels) * 100.0
    cached_confusion_mask = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2RGB)

    boosted_frame = boost_color_contrast_lab(frame, final_mask)
    cached_simulated_frame = cv2.resize(low_res_sim, (w, h)).astype(np.uint8)

    gray_full = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    laplacian_var = cv2.Laplacian(gray_full, cv2.CV_64F).var()
    current_scale = float(np.clip(0.6 + (laplacian_var * 0.0008), 0.7, 1.4))

    if flow is not None:
        flow_x = flow[..., 0]
        flow_y = flow[..., 1]
        mask_resized = cv2.resize(final_mask, (flow_x.shape[1], flow_x.shape[0])) / 255.0
        mean_dx = np.sum(flow_x * mask_resized) / (np.sum(mask_resized) + 1e-6)
        mean_dy = np.sum(flow_y * mask_resized) / (np.sum(mask_resized) + 1e-6)
        current_angle = float(np.degrees(np.arctan2(mean_dy, mean_dx)))
    else:
        current_angle = 0.0

    angle_factor = 1 + (abs(current_angle) / 180.0)
    current_scale = float(np.clip(current_scale * angle_factor, 0.7, 2.0))
    local_texture = generate_local_clean_texture(texture_dropdown, h, w, current_scale, current_angle, is_live_stream)

    brightness_value = cv2.mean(gray_full, mask=final_mask)[0]
    if brightness_value < 130:
        local_texture = cv2.bitwise_not(local_texture)
    else:
        local_texture = cv2.threshold(local_texture, 200, 255, cv2.THRESH_BINARY_INV)[1]

    cached_final_composited = apply_alpha_blending(boosted_frame, local_texture, cached_confusion_mask, adaptive_alpha)

    if not is_static_tab:
        real_center_x = int(w / 2)
        real_center_y = int(h / 2 + (np.sin(time.time() * 2.5) * (h / 240 * 35) if is_live_stream else 0))
        text_pos = (max(0, min(w - 1, real_center_x + motion_x)), max(0, min(h - 1, real_center_y + motion_y)))

        cv2.putText(cached_final_composited, tip_text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2,
                    cv2.LINE_AA)

        if show_foveated_circle:
            real_focus_radius = int(min(w, h) * 0.45)
            cv2.circle(cached_final_composited, (real_center_x, real_center_y), real_focus_radius, (0, 255, 0), 2,
                       lineType=cv2.LINE_AA)

    full_status_report = (
        f"{status_text}\n"
        f"⚡ Performance Metrics: Execution Speed = {fps_current_rate:.1f} FPS\n"
        f"🎯 Matrix Output Block: Dynamic Grid Allocation = {w}x{h} | Confusion Density = {confusion_percentage:.1f}%\n"
        f"🌊 Motion State Matrix Tracker: Current Class = {tip_text}\n"
        f"📐 Spatial HUD Tracking: Dynamic Fovea Movement Enabled | Scale = {current_scale:.2f}x\n"
        f"⚙️ Sensory Feed: Ambient Lighting Coefficient = {mean_brightness:.1f}"
    )
    return frame, cached_simulated_frame, cached_confusion_mask, cached_final_composited, full_status_report


def load_initial_preview(texture_dropdown, show_foveated_circle=True):
    img, sim, mask, comp, report = process_universal_pipeline(None, texture_dropdown, show_foveated_circle)
    return img, sim, mask, comp


# -------------------------------------------------------------------------
# GLOBAL HELPER FUNCTIONS FOR BACKEND CONTROLS
# -------------------------------------------------------------------------
ISHIHARA_REGISTRY = {
    "https://upload.wikimedia.org/wikipedia/commons/e/e0/Ishihara_9.png": "74",
    "https://upload.wikimedia.org/wikipedia/commons/b/b5/Ishihara_11.png": "6",
    "https://upload.wikimedia.org/wikipedia/commons/2/22/Ishihara_23.jpg": "42"
}


def get_random_ishihara_plate():
    import random
    url = random.choice(list(ISHIHARA_REGISTRY.keys()))
    answer = ISHIHARA_REGISTRY[url]
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=5)
        matrix = np.array(Image.open(io.BytesIO(response.content)))
        return matrix, answer
    except Exception:
        fallback = np.zeros((300, 300, 3), dtype=np.uint8)
        rand_num = str(random.choice([12, 42, 73, 6, 29]))
        cv2.circle(fallback, (150, 150), 130, (70, 180, 70), -1)
        cv2.putText(fallback, rand_num, (95, 175), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (50, 50, 180), 5)
        return fallback, rand_num


def check_ishihara_answer(user_input, correct_answer):
    clean_input = str(user_input).strip()

    if clean_input == str(correct_answer):
        return (
            "✨ Your vision is normal. Assessment Complete.",
            gr.Textbox(value=clean_input, interactive=False),
            gr.Button(visible=False),
            gr.Markdown("### ✅ Verification Token: VALIDATED", visible=True),
            gr.Image(),
            correct_answer,
            gr.Button(visible=False),
            gr.Button(visible=True)
        )
    else:
        new_plate_matrix, new_answer = get_random_ishihara_plate()
        error_msg = f"⚠️ You entered '{clean_input}'. That is incorrect. The target plate was shuffled! Try to solve this new pattern."
        return (
            error_msg,
            gr.Textbox(value="", placeholder="Type the new number here...", interactive=True),
            gr.Button(visible=True),
            gr.Markdown(visible=False),
            new_plate_matrix,
            new_answer,
            gr.Button(visible=False),
            gr.Button(visible=True)
        )


def calculate_cvd_probability(ans1, ans2, ans3):
    score = 0
    if ans1 in ["Heavily saturated Red", "Heavily saturated Green"]:
        score += 30
    elif ans1 == "Looks identical across most of the slider.":
        score += 35

    if ans2 == "No pattern is visible.":
        score += 35
    elif ans2 == "73 (Alternative axis contrast)":
        score += 20

    if ans3 == "It fades completely into the dark background":
        score += 30
    elif ans3 == "It dims slightly but keeps a clear, visible edge":
        score += 10

    prob = min(score, 100)
    status = f"The probability of developing color blindness is {{{prob}%}}."
    return prob, status


# التعديل: دالة لتفريغ وإعادة تهيئة حقول التقييم المتقدم للحالة الافتراضية
def reset_advanced_workspace():
    return (
        gr.Radio(value="Balanced standard mix"),
        gr.Radio(value="42 (Standard contrast)"),
        gr.Radio(value="It remains sharp and vivid against the darkness"),
        0,
        "Formulate metrics to calculate probabilities."
    )


def save_live_snapshot():
    global cached_final_composited
    if cached_final_composited is not None:
        return Image.fromarray(cached_final_composited)
    return None


def export_telemetry_report(status_report):
    report_path = "chromasight_telemetry_report.txt"
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("===================================================\n")
            f.write("     CHROMASIGHT AI - COGNITIVE TELEMETRY          \n")
            f.write("===================================================\n\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(status_report)
        return report_path
    except Exception as e:
        print(f"Error compiling telemetry log file: {e}")
        return None


def reset_diagnostic_workspace():
    new_plate_matrix, new_answer = get_random_ishihara_plate()
    return (
        new_plate_matrix,
        "",
        "Awaiting user input matrix...",
        gr.Textbox(interactive=True),
        gr.Button(visible=False),
        gr.Markdown(visible=False),
        new_answer
    )


def export_diagnostic_report(feedback_matrix, current_guess, correct_answer):
    report_path = "chromasight_diagnostic_result.txt"
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("===================================================\n")
            f.write("     CHROMASIGHT AI - ISHIHARA DIAGNOSTIC LOG      \n")
            f.write("===================================================\n\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Evaluation Status: {feedback_matrix}\n")
        return report_path
    except Exception as e:
        print(f"Error compiling diagnostic log file: {e}")
        return None


# -------------------------------------------------------------------------
# MODULE 1: UI Layout Dashboard
# -------------------------------------------------------------------------
accessible_css = """
    #accessible_blue_btn {
        background-color: #0055ff !important;
        color: #ffffff !important;            
        border: 2px solid #ffffff !important;
        font-weight: bold !important;
        box-shadow: 0 4px 6px rgba(0, 85, 255, 0.2);
    }
    #accessible_blue_btn:hover {
        background-color: #0044cc !important; 
    }

    .cvd_friendly_panel {
        font-family: 'Segoe UI', Arial, sans-serif !important;
        color: #002244 !important;
        font-size: 16px !important;
        background-color: #FAFAFA !important;
        border-left: 6px solid #0072B2 !important;
        padding: 15px !important;
        border-radius: 4px !important;
    }

    #cvd_advanced_assessment_btn {
        background-color: #0072B2 !important; 
        color: #ffffff !important;
        font-weight: bold !important;
        border: 2px solid #F0E442 !important;
        font-size: 14px !important;
    }
"""

init_orig, init_sim, init_mask, init_final = load_initial_preview("dots", True)

with gr.Blocks(title="ChromaSight AI - Absolute Solution") as demo:
    print("👋 Welcome to ChromaSight AI Dashboard Configuration Engine!")
    print("👉 If you want to access the color chroma sight interface features,")
    print("   please click the local URL link provided above.")
    print("=" * 60 + "\n")

    gr.Markdown(
        """
        # 👁️ ChromaSight AI — High-Fidelity Foveated Cognitive Assistive Engine
        ### Telemetry-Driven Adaptive Vision System
        """
    )

    with gr.Tabs() as main_tabs_container:
        # -----------------------------------------------------------------
        # TAB 1: LIVE WEBCAM STREAM MODE
        # -----------------------------------------------------------------
        with gr.Tab("🎥 Live Webcam Stream", id=0):
            gr.HTML(
                """<div style="padding: 12px; border-radius: 10px; background-color: #f5f5f5; border: 1px solid #ddd; font-size: 14px; line-height: 1.6;"><b>System Output Guide:</b><br><br><b>Image 1:</b> Original live camera feed.<br><b>Image 2:</b> Deuteranopia simulation.<br><b>Image 3:</b> Confusion mask.<br><b>Image 4:</b> AI-enhanced output.</div>""")
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 🛠️ System Controls")
                    webcam_input = gr.Image(sources=["webcam"], type="numpy", streaming=True,
                                            label="Active Camera Feed")
                    texture_dropdown = gr.Dropdown(choices=["dots", "hatching", "voronoi"], value="dots",
                                                   label="Texture Style")
                    show_fovea_checkbox = gr.Checkbox(value=True, label="Show Foveated Circle")
                    snapshot_btn = gr.Button("📸 Take Snapshot", variant="primary")
                    snapshot_download = gr.Image(label="Saved Snapshot", interactive=False, type="pil")

                    scene_telemetry = gr.Textbox(label="Cognitive Telemetry HUD", interactive=False, lines=6)
                    telemetry_download_btn = gr.Button("📄 Download Telemetry Report", variant="secondary")
                    telemetry_file_output = gr.File(label="Generated Log File", value=None, visible=True)

                    snapshot_btn.click(fn=save_live_snapshot, inputs=[], outputs=snapshot_download)
                    telemetry_download_btn.click(fn=export_telemetry_report, inputs=[scene_telemetry],
                                                 outputs=[telemetry_file_output])

                with gr.Column(scale=3):
                    gr.Markdown("### 📊 Live Processing Output")
                    with gr.Row():
                        out_orig = gr.Image(value=init_orig, label="Original Frame", interactive=False)
                        out_sim = gr.Image(value=init_sim, label="CVD Simulation", interactive=False)
                    with gr.Row():
                        out_mask = gr.Image(value=init_mask, label="Confusion Mask", interactive=False)
                        out_final = gr.Image(value=init_final, label="Final Enhanced Output", interactive=False)

            webcam_input.stream(
                fn=process_universal_pipeline,
                inputs=[webcam_input, texture_dropdown, show_fovea_checkbox],
                outputs=[out_orig, out_sim, out_mask, out_final, scene_telemetry]
            )

        # -----------------------------------------------------------------
        # TAB 2: STATIC IMAGE PROCESSING MODE
        # -----------------------------------------------------------------
        with gr.Tab("🖼️ Static Image Upload", id=1):
            gr.HTML(
                """<div style="padding: 12px; border-radius: 10px; background-color: #f5f5f5; border: 1px solid #ddd; font-size: 14px; line-height: 1.6;"><b>System Output Guide:</b><br><br><b>Image 1:</b> Original input image.<br><b>Image 2:</b> Deuteranopia simulation.<br><b>Image 3:</b> Confusion mask.<br><b>Image 4:</b> AI-enhanced output.</div>""")

            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.Image(type="numpy", label="Upload Image")
                    texture_dropdown_2 = gr.Dropdown(choices=["dots", "hatching", "voronoi"], value="dots",
                                                     label="Texture Style")
                    static_btn = gr.Button("Process Image")

                with gr.Column(scale=3):
                    with gr.Row():
                        out_orig_static = gr.Image(label="Original", interactive=False)
                        out_sim_static = gr.Image(label="Simulation", interactive=False)
                    with gr.Row():
                        out_mask_static = gr.Image(label="Mask", interactive=False)
                        out_final_static = gr.Image(label="Final Output", interactive=False)

            static_btn.click(
                fn=lambda img, tex: process_universal_pipeline(img, tex, show_foveated_circle=False,
                                                               is_static_tab=True),
                inputs=[image_input, texture_dropdown_2],
                outputs=[out_orig_static, out_sim_static, out_mask_static, out_final_static, scene_telemetry]
            )

        # -----------------------------------------------------------------
        # TAB 3: SHUFFLED ISHIHARA DIAGNOSTIC TEST
        # -----------------------------------------------------------------
        with gr.Tab("👁️ Diagnostic Test", id=2):
            gr.HTML(
                """<div style="padding: 12px; border-radius: 10px; background-color: #f5f5f5; border: 1px solid #ddd; font-size: 14px; line-height: 1.6; text-align: center;"><b>Ishihara Screening Plate</b><br>Identify the hidden numeral embedded within the colored dots.</div>""")

            initial_plate, initial_answer = get_random_ishihara_plate()
            hidden_answer_key = gr.State(value=initial_answer)

            with gr.Group() as ishihara_core_view:
                with gr.Row():
                    with gr.Column(scale=2):
                        ishihara_img_component = gr.Image(value=initial_plate, label="Ishihara Test Plate",
                                                          interactive=False)

                    with gr.Column(scale=1):
                        gr.Markdown("### 🎛️ Evaluation Panel")
                        user_guess = gr.Textbox(label="What number do you see inside the circle?",
                                                placeholder="Type the number here...", max_lines=1)

                        with gr.Row():
                            diagnostic_btn = gr.Button("Evaluate Vision Profile", variant="primary")
                            reset_diagnostic_btn = gr.Button("🔄 Skip Pattern", variant="secondary", visible=False)

                        advanced_assessment_trigger = gr.Button("Advanced Color Vision Assessment",
                                                                elem_id="cvd_advanced_assessment_btn", visible=False)
                        cvd_assistance_btn = gr.Button("🔍 Activate ChromaSight Vision Boosters", variant="stop",
                                                       visible=False)
                        success_token = gr.Markdown("### ✅ Verification Token: VALIDATED", visible=False)

                        diagnostic_results = gr.Label(label="Diagnostic Feedback Matrix",
                                                      value="Awaiting user input matrix...")

                        export_diagnostic_btn = gr.Button("💾 Save Diagnostic Result", variant="secondary")
                        diagnostic_file_output = gr.File(label="Generated Diagnostic Log", value=None, visible=True)

            # لوحة الاختبار المتقدم
            with gr.Group(visible=False) as advanced_assessment_panel:
                with gr.Group(elem_classes="cvd_friendly_panel"):
                    gr.Markdown("## 📑 Advanced Color Vision Assessment Profile")
                    gr.Markdown(
                        "This dynamic framework isolates sub-clinical chromatic boundaries and spectral luminous efficiencies.")

                    q1 = gr.Radio(
                        label="Question 1: Red-Green Balance (Rayleigh Match)\nIdentifies green-weakness (Deuteranomaly) or red-weakness (Protanomaly) based on the light ratio needed to match a pure yellow reference.",
                        choices=["Balanced standard mix", "Heavily saturated Red", "Heavily saturated Green",
                                 "Looks identical across most of the slider."],
                        value="Balanced standard mix"
                    )
                    gr.Markdown("---")

                    q2 = gr.Radio(
                        label="Question 2: Isolate Chromatic Boundaries (Luminance Equalization)\nTests color confusion lines by stripping away brightness (luminance) differences, forcing the eye to rely purely on chromatic contrast.",
                        choices=["42 (Standard contrast)", "73 (Alternative axis contrast)", "12 (Universal control)",
                                 "No pattern is visible."],
                        value="42 (Standard contrast)"
                    )
                    gr.Markdown("---")

                    q3 = gr.Radio(
                        label="Question 3: Red Dimming Threshold (Spectral Luminous Efficiency)\nIdentifies red-blindness (Protanopia). Individuals with this profile perceive long-wavelength deep reds as significantly darker or black compared to normal trichromats.",
                        choices=["It fades completely into the dark background",
                                 "It dims slightly but keeps a clear, visible edge",
                                 "It remains sharp and vivid against the darkness"],
                        value="It remains sharp and vivid against the darkness"
                    )

                    with gr.Row():
                        calculate_btn = gr.Button("Calculate CVD Probability Profile", variant="primary")
                        # التعديل: زر إعادة تهيئة التقييم المتقدم بدون إنعاش الصفحة
                        reset_advanced_btn = gr.Button("🔄 Reset Assessment", variant="secondary")

                    cvd_probability_bar = gr.Slider(minimum=0, maximum=100, step=1,
                                                    label="CVD Risk Probability Coefficient (%)", interactive=False)
                    cvd_probability_text = gr.Label(label="Diagnostic Interpretation Output",
                                                    value="Formulate metrics to calculate probabilities.")

                    close_advanced_btn = gr.Button("↩️ Return to Core Test", variant="secondary")

            diagnostic_btn.click(
                fn=check_ishihara_answer,
                inputs=[user_guess, hidden_answer_key],
                outputs=[diagnostic_results, user_guess, reset_diagnostic_btn, success_token,
                         ishihara_img_component, hidden_answer_key, cvd_assistance_btn, advanced_assessment_trigger]
            )

            reset_diagnostic_btn.click(
                fn=reset_diagnostic_workspace,
                inputs=[],
                outputs=[ishihara_img_component, user_guess, diagnostic_results, user_guess,
                         reset_diagnostic_btn, success_token, hidden_answer_key]
            ).then(fn=lambda: (gr.Button(visible=False), gr.Button(visible=False)),
                   outputs=[cvd_assistance_btn, advanced_assessment_trigger])

            export_diagnostic_btn.click(
                fn=export_diagnostic_report,
                inputs=[diagnostic_results, user_guess, hidden_answer_key],
                outputs=[diagnostic_file_output]
            )

            calculate_btn.click(
                fn=calculate_cvd_probability,
                inputs=[q1, q2, q3],
                outputs=[cvd_probability_bar, cvd_probability_text]
            )

            # التعديل: ربط حدث زر التصفير المتقدم بالمكونات التفاعلية للـ Advanced Tab
            reset_advanced_btn.click(
                fn=reset_advanced_workspace,
                inputs=[],
                outputs=[q1, q2, q3, cvd_probability_bar, cvd_probability_text]
            )

            advanced_assessment_trigger.click(
                fn=lambda: (gr.Group(visible=False), gr.Group(visible=True)),
                inputs=[],
                outputs=[ishihara_core_view, advanced_assessment_panel]
            )

            close_advanced_btn.click(
                fn=lambda: (gr.Group(visible=True), gr.Group(visible=False)),
                inputs=[],
                outputs=[ishihara_core_view, advanced_assessment_panel]
            )

if __name__ == "__main__":
    demo.queue().launch(css=accessible_css)