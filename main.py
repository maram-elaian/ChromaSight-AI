#--------------------------------------------------------------------------
import io
import cv2
import requests
import numpy as np
import gradio as gr
from PIL import Image

# -------------------------------------------------------------------------
# MODULE 2 CORE: Computer Vision Algorithms
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


def get_confusion_mask(original, simulated, threshold=25):
    orig_lab = cv2.cvtColor(original, cv2.COLOR_RGB2LAB)
    sim_lab = cv2.cvtColor(simulated, cv2.COLOR_RGB2LAB)

    orig_a, orig_b = orig_lab[:, :, 1], orig_lab[:, :, 2]
    sim_a, sim_b = sim_lab[:, :, 1], sim_lab[:, :, 2]

    diff_a = cv2.absdiff(orig_a, sim_a)
    diff_b = cv2.absdiff(orig_b, sim_b)
    diff = cv2.addWeighted(diff_a, 0.5, diff_b, 0.5, 0)

    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)

    return cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)


# -------------------------------------------------------------------------
# MODULE 3: Generative AI Inpainting Engine (الذكاء الاصطناعي التوليدي)
# -------------------------------------------------------------------------
# ملاحظة للمناقشة: ضعي مفتاح الـ Token الخاص بكِ من موقع Hugging Face هنا ليعمل الاتصال السحابي
HF_API_TOKEN = "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"
API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}


def numpy_to_bytes(img_array):
    """تحويل المصفوفة البرمجية إلى بايتات مضغوطة لإرسالها عبر الإنترنت للموديل"""
    success, encoded_image = cv2.imencode('.png', img_array)
    if not success:
        raise ValueError("Could not encode image to PNG.")
    return encoded_image.tobytes()


def generate_ai_texture_inpainting(original_rgb, binary_mask, texture_style="dots"):
    """ترسل الصورة والقناع إلى السحابة لملء الفراغات بأنماط هندسية مخصصة"""
    prompt_mapping = {
        "dots": "Clean minimalist monochrome micro-dot matrix pattern, uniform spacing, high contrast, fine vector lines",
        "hatching": "Highly detailed fine monochrome cross-hatching geometric line pattern, crisp clean thin strokes",
        "voronoi": "Organic Voronoi diagram cell pattern, ultra-fine monochrome continuous lines"
    }

    target_prompt = prompt_mapping.get(texture_style, prompt_mapping["dots"])

    try:
        # تجهيز البيانات للإرسال
        image_bytes = numpy_to_bytes(original_rgb)
        mask_bytes = numpy_to_bytes(binary_mask)

        payload = {
            "inputs": {
                "image": image_bytes,
                "mask_image": mask_bytes,
                "prompt": target_prompt,
                "negative_prompt": "blurry, colorful, ugly, chaotic, gradients, photorealistic background",
                "num_inference_steps": 4,  # خطوات قليلة جداً لضمان السرعة الفائقة
                "guidance_scale": 1.5
            }
        }

        # الاتصال بالخادم السحابي
        response = requests.post(API_URL, headers=headers, json=payload, timeout=10)

        if response.status_code == 200:
            generated_image = Image.open(io.BytesIO(response.content)).convert("RGB")
            return np.array(generated_image)
        else:
            print(f"API Warning ({response.status_code}): Using fallback visual.")
            return None
    except Exception as e:
        print(f"Connection skipped or failed: {e}")
        return None


# -------------------------------------------------------------------------
# INTERMEDIARY PIPELINE FUNCTION (دالة الربط المركزي)
# -------------------------------------------------------------------------
def process_static_image(frame, threshold_slider, texture_dropdown):
    if frame is None:
        return None, None, None, None

    # 1. توليد محاكاة عمى الألوان
    simulated_frame = simulate_deuteranopia(frame)

    # 2. استخراج قناع الالتباس البصري
    confusion_mask = get_confusion_mask(frame, simulated_frame, threshold=threshold_slider)

    # 3. استدعاء محرك الذكاء الاصطناعي لملء القناع بالنمط المختار
    gray_mask = confusion_mask[:, :, 0]  # الموديل يتطلب قناع من قناة واحدة (رمادي)
    ai_textured_output = generate_ai_texture_inpainting(frame, gray_mask, texture_style=texture_dropdown)

    # في حال لم يتم وضع التوكن أو فشل الاتصال، تعرض الصورة الأصلية كحماية من التوقف
    if ai_textured_output is None:
        ai_textured_output = frame

    # نُعيد الآن 4 مخرجات لعرضها في الـ 4 شاشات بالواجهة
    return frame, simulated_frame, confusion_mask, ai_textured_output


# -------------------------------------------------------------------------
# MODULE 1: Gradio UI Layout Dashboard (تصميم الواجهة)
# -------------------------------------------------------------------------
with gr.Blocks(title="ChromaSight AI - Dashboard") as demo:
    gr.Markdown(
        """
        # 👁️ ChromaSight AI — Diagnostic & Generative Visualization Dashboard
        ### Core Academic Prototype: Computer Vision + Cloud GenAI Inpainting Pipeline
        """
    )

    with gr.Row():
        # لوحة التحكم الجانبية (يسار)
        with gr.Column(scale=1):
            gr.Markdown("### 🛠️ System Configurations")

            image_input = gr.Image(
                value="test.jpg",
                type="numpy",
                label="Input Image (test.jpg)",
                show_label=True
            )

            cvd_type = gr.Dropdown(
                choices=["Deuteranopia (Green-Blind)"],
                value="Deuteranopia (Green-Blind)",
                label="CVD Classification Profile"
            )

            # قائمة جديدة لاختيار شكل النقوش التوليدية حياً
            texture_dropdown = gr.Dropdown(
                choices=["dots", "hatching", "voronoi"],
                value="dots",
                label="AI Texture Pattern Style"
            )

            threshold_slider = gr.Slider(
                minimum=5,
                maximum=60,
                value=25,
                step=1,
                label="Delta-E Sensitivity Threshold"
            )

        # لوحة العرض الرباعية (يمين) - أضفنا الشاشة الرابعة للذكاء الاصطناعي
        with gr.Column(scale=3):
            gr.Markdown("### 📊 Real-Time Pipeline Displays")

            with gr.Row():
                out_orig = gr.Image(label="1. Original Image", interactive=False)
                out_sim = gr.Image(label="2. Perceptual CVD Simulation", interactive=False)

            with gr.Row():
                out_mask = gr.Image(label="3. Extracted Confusion Mask", interactive=False)
                out_ai = gr.Image(label="4. GenAI Textured Inpainting Output", interactive=False)

    # --- ربط أحداث التحديث التلقائي الفوري ---
    inputs_list = [image_input, threshold_slider, texture_dropdown]
    outputs_list = [out_orig, out_sim, out_mask, out_ai]

    demo.load(fn=process_static_image, inputs=inputs_list, outputs=outputs_list)
    threshold_slider.change(fn=process_static_image, inputs=inputs_list, outputs=outputs_list)
    texture_dropdown.change(fn=process_static_image, inputs=inputs_list, outputs=outputs_list)
    image_input.change(fn=process_static_image, inputs=inputs_list, outputs=outputs_list)

if __name__ == "__main__":
    # تشغيل المشروع بالثيم الناعم والأنيق المعتمد أكاديمياً
    demo.launch(theme=gr.themes.Soft(), share=False)