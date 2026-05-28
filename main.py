import io
import cv2
import requests
import numpy as np
import gradio as gr
from PIL import Image

# -------------------------------------------------------------------------
# MODULE 2 CORE: Computer Vision Algorithms (محاكاة عمى الألوان والقناع)
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


def get_confusion_mask(original, simulated, threshold=25, min_area_size=500):
    """
    نسخة مطورة برؤية كلاسيكية: تستخرج القناع ثم تعزل الأجسام المتماسكة
    وتحذف النويز والبكسلات العشوائية بناءً على المساحة (Connected Components).
    """
    # 1. الحسابات التقليدية لفضاء LAB واستخراج الفروقات
    orig_lab = cv2.cvtColor(original, cv2.COLOR_RGB2LAB)
    sim_lab = cv2.cvtColor(simulated, cv2.COLOR_RGB2LAB)

    orig_a, orig_b = orig_lab[:, :, 1], orig_lab[:, :, 2]
    sim_a, sim_b = sim_lab[:, :, 1], sim_lab[:, :, 2]

    diff_a = cv2.absdiff(orig_a, sim_a)
    diff_b = cv2.absdiff(orig_b, sim_b)
    diff = cv2.addWeighted(diff_a, 0.5, diff_b, 0.5, 0)

    # 2. تحويل الفروقات إلى قناع ثنائي (أبيض وأسود)
    _, raw_mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    # 3. خطوة السحر الكلاسيكي: تحليل المكونات المتصلة لعزل الأقاليم
    # num_labels: عدد الأجسام المكتشفة
    # labels_im: مصفوفة تشبه الصورة ولكن كل جسم له رقم (الجسم الأول بكسلاته تحمل رقم 1، الثاني رقم 2 وهكذا)
    # stats: تحتوي على معلومات كل جسم (بما في ذلك مساحته بالبكسل في العامود الأخير cv2.CC_STAT_AREA)
    num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(raw_mask)

    # إنشاء قناع جديد نظيف وفارغ تماماً وضخ الكتل المتماسكة داخله
    cleaned_mask = np.zeros_like(raw_mask)

    # المرور على كل الأجسام المكتشفة (نبدأ من 1 لأن الرقم 0 هو الخلفية السوداء دائماً)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]  # استخراج مساحة الجسم الحالي بالبكسل

        # شرط تماسك الكائن (Object Consistency):
        # إذا كانت مساحة الكتلة البيضاء أكبر من الحجم المحدد (مثلاً 500 بكسل)، نعتبرها كائناً حقيقياً ونبقي عليها
        if area >= min_area_size:
            # دمج هذا الكائن المتصل داخل القناع النظيف
            cleaned_mask[labels_im == i] = 255

    # 4. تحسين الحواف للأجسام المتبقية لجعلها ناعمة ومتناسقة للذكاء الاصطناعي
    kernel = np.ones((5, 5), np.uint8)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)
    cleaned_mask = cv2.dilate(cleaned_mask, kernel, iterations=1)

    # إرجاع القناع المطور على مستوى الأقاليم كـ 3 قنوات للواجهة
    return cv2.cvtColor(cleaned_mask, cv2.COLOR_GRAY2RGB)

# -------------------------------------------------------------------------
# MODULE 3: Generative AI Inpainting Engine (الاتصال بالذكاء الاصطناعي)
# -------------------------------------------------------------------------
# تذكير: ضعي مفتاح التوكن الخاص بكِ هنا بين علامتي التنصيص ليعمل الاتصال السحابي
HF_API_TOKEN = "your_hf_token_here"
API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}


def numpy_to_bytes(img_array):
    success, encoded_image = cv2.imencode('.png', img_array)
    if not success:
        raise ValueError("Could not encode image to PNG.")
    return encoded_image.tobytes()


def generate_ai_texture_inpainting(original_rgb, binary_mask, texture_style="dots"):
    prompt_mapping = {
    "dots": "Bold black and white geometric polka dot pattern, thick dark lines, solid high-contrast micro-dots, vector grid, sharp edges",
    "hatching": "Coarse thick monochrome cross-hatching lines, high contrast black and white blueprint texture, sharp geometric strokes",
    "voronoi": "Sharp black and white Voronoi tessellation cells, thick high-contrast dark continuous borders"
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
        else:
            print(f"API Warning ({response.status_code}): Using fallback visual.")
            return None
    except Exception as e:
        print(f"Connection skipped or failed: {e}")
        return None


# -------------------------------------------------------------------------
# MODULE 4: Matrix Fusion & Alpha Blending (كود الدمج الرياضي الشفاف الجديد)
# -------------------------------------------------------------------------
def apply_alpha_blending(original_rgb, ai_textured_rgb, binary_mask_rgb, alpha=0.35):
    """تقوم بدمج أنماط الذكاء الاصطناعي فوق الصورة الأصلية فقط داخل حدود القناع الأبيض وبشفافية ناعمة"""
    # التأكد من تطابق مقاسات الصور والقناع
    if original_rgb.shape != ai_textured_rgb.shape or original_rgb.shape != binary_mask_rgb.shape:
        ai_textured_rgb = cv2.resize(ai_textured_rgb, (original_rgb.shape[1], original_rgb.shape[0]))
        binary_mask_rgb = cv2.resize(binary_mask_rgb, (original_rgb.shape[1], original_rgb.shape[0]))

    # تحويل المصفوفات إلى float32 لمنع حدوث أخطاء حسابية أثناء الضرب والجمع
    img_orig = original_rgb.astype(np.float32)
    img_text = ai_textured_rgb.astype(np.float32)

    # مصفوفة القناع العادي (تُحور القيم من 0-255 إلى نطاق الشفافية المطلوب 0.0 - 0.35)
    normalized_mask = (binary_mask_rgb.astype(np.float32) / 255.0) * alpha

    # تطبيق معادلة الدمج الرياضية: Final = (1 - Mask) * Original + Mask * Texture
    blended_output = (1.0 - normalized_mask) * img_orig + normalized_mask * img_text

    # إعادة القيم إلى النطاق الطبيعي للصورة 0-255 وتحويلها إلى نوع uint8
    return np.clip(blended_output, 0, 255).astype(np.uint8)


# -------------------------------------------------------------------------
# INTERMEDIARY PIPELINE FUNCTION (الدالة الوسيطة للربط بين المراحل الأربعة)
# -------------------------------------------------------------------------
def process_static_image(frame, threshold_slider, texture_dropdown):
    if frame is None:
        return None, None, None, None

    # المرحلة 2: محاكاة تجربة عمى الألوان
    simulated_frame = simulate_deuteranopia(frame)

    # المرحلة 2 أيضاً: استخراج قناع المناطق الملتبسة
    confusion_mask = get_confusion_mask(frame, simulated_frame, threshold=threshold_slider)

    # المرحلة 3: توليد النقوش عبر الذكاء الاصطناعي السحابي
    gray_mask = confusion_mask[:, :, 0]
    ai_textured_output = generate_ai_texture_inpainting(frame, gray_mask, texture_style=texture_dropdown)

    # حماية الكود: إذا كان التوكن خاطئاً أو لم يتصل بالسيرفر، نعتبر مخرجات الذكاء الاصطناعي هي نفس الصورة
    if ai_textured_output is None:
        ai_textured_output = frame

    # المرحلة 4 (الجديدة): دمج النقوش فوق الصورة الأصلية بأسلوب الشفافية الذكية
        # تعديل السطر في دالة process_static_image لزيادة قوة ظهور النمط:
        final_composited_output = apply_alpha_blending(
            original_rgb=frame,
            ai_textured_rgb=ai_textured_output,
            binary_mask_rgb=confusion_mask,
            alpha=0.65  # رفع الشفافية يجعل خطوط ونقاط الذكاء الاصطناعي حادة وبارزة جداً
        )

    # إرسال المخرجات الأربعة إلى شاشات العرض الأربعة بالترتيب
    return frame, simulated_frame, confusion_mask, final_composited_output


# -------------------------------------------------------------------------
# MODULE 1: Gradio UI Layout Dashboard (الواجهة التفاعلية)
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

        # لوحة العرض الرباعية (يمين)
        with gr.Column(scale=3):
            gr.Markdown("### 📊 Real-Time Pipeline Displays")

            with gr.Row():
                out_orig = gr.Image(label="1. Original Image", interactive=False)
                out_sim = gr.Image(label="2. Perceptual CVD Simulation", interactive=False)

            with gr.Row():
                out_mask = gr.Image(label="3. Extracted Confusion Mask", interactive=False)
                # الشاشة الرابعة تعرض الآن النتيجة النهائية المدمجة بوضوح احترافي
                out_final = gr.Image(label="4. Final Composited Accessibility Output (Module 4)", interactive=False)

    # ربط أحداث التحديث التلقائي الفوري عند أي تغيير بالمتصفح
    inputs_list = [image_input, threshold_slider, texture_dropdown]
    outputs_list = [out_orig, out_sim, out_mask, out_final]

    demo.load(fn=process_static_image, inputs=inputs_list, outputs=outputs_list)
    threshold_slider.change(fn=process_static_image, inputs=inputs_list, outputs=outputs_list)
    texture_dropdown.change(fn=process_static_image, inputs=inputs_list, outputs=outputs_list)
    image_input.change(fn=process_static_image, inputs=inputs_list, outputs=outputs_list)

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(), share=False)