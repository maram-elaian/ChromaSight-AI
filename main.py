import io
import cv2
import requests
import numpy as np
import gradio as gr
from PIL import Image

# -------------------------------------------------------------------------
# MODULE 2: Advanced Region-Based Vision Filters
# -------------------------------------------------------------------------
M = np.array([
    [0.430, 0.720, -0.150],
    [0.340, 0.620, 0.040],
    [-0.020, 0.030, 0.990]
], dtype=np.float32)


def simulate_deuteranopia(frame):
    img = frame.astype(np.float32) / 255.0
    out = cv2.transform(img, M)
    out = np.clip(out, 0, 1)
    out = (out * 255).astype(np.uint8)
    return out


# -------------------------------------------------------------------------
# MODULE 3: Generative AI Cloud Inpainting Engine
# -------------------------------------------------------------------------
HF_API_TOKEN = "your_hf_token_here"  # ضعي التوكن الخاص بكِ هنا
API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}


def numpy_to_bytes(img_array):
    success, encoded_image = cv2.imencode('.png', img_array)
    if not success:
        raise ValueError("Could not encode image to PNG.")
    return encoded_image.tobytes()


def generate_ai_texture_inpainting(original_rgb, binary_mask, texture_style):
    """يستقبل النمط الذي تم اختياره تلقائياً لكل إقليم ويولده سحابياً"""
    prompt_mapping = {
        "dots": "Clean minimalist monochrome micro-dot matrix pattern, uniform spacing, high contrast, fine vector lines",
        "hatching": "Highly detailed fine monochrome cross-hatching geometric line pattern, crisp clean thin strokes",
        "voronoi": "Organic Voronoi diagram cell pattern, ultra-fine monochrome continuous lines",
        "wireframe": "Minimalist isometric wireframe grid pattern, thin sharp continuous lines"
    }

    target_prompt = prompt_mapping.get(texture_style, prompt_mapping["dots"])

    try:
        image_bytes = numpy_to_bytes(original_rgb)
        mask_bytes = numpy_to_bytes(binary_mask)

        payload = {
            "inputs": {
                "image": image_bytes,
                "mask_image": mask_bytes,
                "prompt": target_prompt,
                "negative_prompt": "blurry, colorful, ugly, chaotic, gradients, photorealistic background",
                "num_inference_steps": 4,
                "guidance_scale": 1.5
            }
        }

        response = requests.post(API_URL, headers=headers, json=payload, timeout=10)
        if response.status_code == 200:
            generated_image = Image.open(io.BytesIO(response.content)).convert("RGB")
            return np.array(generated_image)
        return None
    except Exception as e:
        print(f"Cloud Engine Offline: {e}")
        return None


# -------------------------------------------------------------------------
# MODULE 4: Alpha Blending Matrix Fusion
# -------------------------------------------------------------------------
def apply_alpha_blending(original_rgb, ai_textured_rgb, binary_mask_rgb, alpha=0.45):
    if original_rgb.shape != ai_textured_rgb.shape or original_rgb.shape != binary_mask_rgb.shape:
        ai_textured_rgb = cv2.resize(ai_textured_rgb, (original_rgb.shape[1], original_rgb.shape[0]))
        binary_mask_rgb = cv2.resize(binary_mask_rgb, (original_rgb.shape[1], original_rgb.shape[0]))

    img_orig = original_rgb.astype(np.float32)
    img_text = ai_textured_rgb.astype(np.float32)
    normalized_mask = (binary_mask_rgb.astype(np.float32) / 255.0) * alpha
    blended_output = (1.0 - normalized_mask) * img_orig + normalized_mask * img_text
    return np.clip(blended_output, 0, 255).astype(np.uint8)


# -------------------------------------------------------------------------
# CORE AUTOMATION: Shape Analyzer & Dynamic Pipeline
# -------------------------------------------------------------------------
def process_dynamic_semantic_pipeline(frame, threshold_slider):
    if frame is None:
        return None, None, None, None, "No Region Detected"

    # 1. توليد محاكاة عمى الألوان الفورية
    simulated_frame = simulate_deuteranopia(frame)

    # 2. حساب الفروقات اللونية لاستخراج القناع الأولي
    orig_lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
    sim_lab = cv2.cvtColor(simulated_frame, cv2.COLOR_RGB2LAB)
    diff = cv2.addWeighted(cv2.absdiff(orig_lab[:, :, 1], sim_lab[:, :, 1]), 0.5,
                           cv2.absdiff(orig_lab[:, :, 2], sim_lab[:, :, 2]), 0.5, 0)
    _, raw_mask = cv2.threshold(diff, threshold_slider, 255, cv2.THRESH_BINARY)

    # 3. محرك اتخاذ القرار الهندسي (Semantic Shape & Size Analyzer)
    num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(raw_mask)

    # مصفوفة القناع النهائي المصفى ونظام تتبع القرارات للشرح في التقرير
    final_mask = np.zeros_like(raw_mask)
    accumulated_textures = np.zeros_like(frame)
    decision_log = []

    # نحدد قيمة دنيا لحجم الأجسام لحذف النويز العشوائي (مثلاً 400 بكسل)
    MIN_AREA = 400

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        if area >= MIN_AREA:
            # استخراج أبعاد الجسم الحالي (العرض والارتفاع)
            width = stats[i, cv2.CC_STAT_WIDTH]
            height = stats[i, cv2.CC_STAT_HEIGHT]
            aspect_ratio = float(width) / height  # نسبة الاستطالة

            # عزل الجسم الحالي في قناع منفصل مؤقت لتمريره للذكاء الاصطناعي
            single_region_mask = np.zeros_like(raw_mask)
            single_region_mask[labels_im == i] = 255
            final_mask[labels_im == i] = 255

            # تطبيق قواعد اتخاذ القرار (Decision Rules):
            if aspect_ratio > 2.5 or aspect_ratio < 0.4:
                # الكائنات الطولية والنحيفة جداً (مثل الخطوط، الحواف، النصوص)
                chosen_style = "hatching"
            elif area > 8000:
                # الكائنات الضخمة جداً والمساحات العريضة (مثل البتلات الكبيرة)
                chosen_style = "voronoi" if aspect_ratio > 0.8 else "wireframe"
            elif area < 2500:
                # الأقاليم المستديرة الصغيرة والبقع
                chosen_style = "dots"
            else:
                # الأقاليم المتوسطة المتزنة
                chosen_style = "dots"

            decision_log.append(
                f"Region {i}: Area={area}px, Ratio={aspect_ratio:.2f} -> Selected: [{chosen_style.upper()}]")

            # استدعاء الذكاء الاصطناعي لمعالجة هذا الإقليم المحدد بالنمط المختار تلقائياً
            textured_region = generate_ai_texture_inpainting(frame, single_region_mask, texture_style=chosen_style)

            if textured_region is not None:
                # دمج الجزء المحدث فقط داخل مصفوفة التجميع الكلية
                idx = (single_region_mask == 255)
                accumulated_textures[idx] = textured_region[idx]

    # في حال فشل الاتصال أو عدم وجود كتل كبيرة، نضع الصورة الأصلية كخلفية للدمج
    if np.sum(accumulated_textures) == 0:
        accumulated_textures = frame

    # 4. دمج مصفوفة الأنماط المجمعة هندسياً فوق الصورة الأصلية بشفافية
    final_mask_rgb = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2RGB)
    final_composited_output = apply_alpha_blending(
        original_rgb=frame,
        ai_textured_rgb=accumulated_textures,
        binary_mask_rgb=final_mask_rgb,
        alpha=0.55
    )

    # تحويل سجل القرارات إلى نص منسق لعرضه أمام المشرفين في الواجهة
    log_text = "\n".join(decision_log) if decision_log else "No valid functional regions met the scale parameters."

    return frame, simulated_frame, final_mask_rgb, final_composited_output, log_text


# -------------------------------------------------------------------------
# MODULE 1: UI Layout Dashboard (الواجهة المحدثة لشاشات التقرير اللحظي)
# -------------------------------------------------------------------------
with gr.Blocks(title="ChromaSight AI - Semantic Dashboard") as demo:
    gr.Markdown(
        """
        # 👁️ ChromaSight AI — Automated Semantic Architecture Dashboard
        ### Advanced Prototype: Shape-Aware Structural Texture Customization Loop
        """
    )

    with gr.Row():
        # الجانب الأيسر للتحكم والمراقبة الذكية
        with gr.Column(scale=1):
            gr.Markdown("### 🛠️ System Configurations")

            image_input = gr.Image(value="test.jpg", type="numpy", label="Input Image (test.jpg)")
            threshold_slider = gr.Slider(minimum=5, maximum=60, value=25, step=1, label="Delta-E Sensitivity Threshold")

            gr.Markdown("### 📜 System Decision Log (سجل قرارات النظام اللحظي)")
            # صندوق نصي تفاعلي يظهر للجنة المناقشة كيف يفكر الكود ويختار الأنماط برمجياً
            output_log = gr.Textbox(label="Automated Assignment Telemetry", interactive=False, lines=10)

        # الجانب الأيمن للعرض الرباعي
        with gr.Column(scale=3):
            gr.Markdown("### 📊 Real-Time Pipeline Displays")

            with gr.Row():
                out_orig = gr.Image(label="1. Original Image", interactive=False)
                out_sim = gr.Image(label="2. Perceptual CVD Simulation", interactive=False)

            with gr.Row():
                out_mask = gr.Image(label="3. Geometrically Filtered Mask", interactive=False)
                out_final = gr.Image(label="4. Semantically Textured Output (Module 4)", interactive=False)

    # ربط الأحداث للتحديث الفوري التلقائي
    inputs_list = [image_input, threshold_slider]
    outputs_list = [out_orig, out_sim, out_mask, out_final, output_log]

    demo.load(fn=process_dynamic_semantic_pipeline, inputs=inputs_list, outputs=outputs_list)
    threshold_slider.change(fn=process_dynamic_semantic_pipeline, inputs=inputs_list, outputs=outputs_list)
    image_input.change(fn=process_dynamic_semantic_pipeline, inputs=inputs_list, outputs=outputs_list)

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(), share=False)