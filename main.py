import cv2
import numpy as np
import gradio as gr

# -------------------------------------------------------------------------
# MODULE 2 CORE: Computer Vision Algorithms (Your Existing Logic)
# -------------------------------------------------------------------------
M = np.array([
    [0.430, 0.720, -0.150],
    [0.340, 0.620, 0.040],
    [-0.020, 0.030, 0.990]
], dtype=np.float32)


def simulate_deuteranopia(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    out = cv2.transform(img, M)
    out = np.clip(out, 0, 1)
    out = (out * 255).astype(np.uint8)
    # Gradio handles standard RGB arrays internally
    return out


def get_confusion_mask(original, simulated, threshold=25):
    # original and simulated are already RGB arrays from Gradio processing
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

    # Return as 3-channel grayscale for beautiful rendering in UI
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)


# -------------------------------------------------------------------------
# INTERMEDIARY PIPELINE FUNCTION
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# INTERMEDIARY PIPELINE FUNCTION (تعديل نظام ألوان الكاميرا الحية)
# -------------------------------------------------------------------------
def process_live_stream(frame, threshold_slider):
    if frame is None:
        return None, None, None

    # السطر الجديد والمهم لإصلاح اللون الأزرق:
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 1. Generate the simulation
    simulated_frame = simulate_deuteranopia(frame)

    # 2. Extract the confusion mask using the dynamic threshold slider value
    confusion_mask = get_confusion_mask(frame, simulated_frame, threshold=threshold_slider)

    # Return all three states to update their respective UI columns simultaneously
    return frame, simulated_frame, confusion_mask


# -------------------------------------------------------------------------
# MODULE 1: Gradio UI Layout Dashboard
# -------------------------------------------------------------------------
with gr.Blocks( title="ChromaSight AI - Dashboard") as demo:
    # Header Section
    gr.Markdown(
        """
        # 👁️ ChromaSight AI — Diagnostic Visualization Dashboard
        ### Accessibility & Human-Centered AI Core Prototype
        """
    )

    # Split Layout: Left Panel for Controls, Right Panel for Diagnostic Stream
    with gr.Row():
        # Interactive Control Sidebar
        with gr.Column(scale=1):
            gr.Markdown("### 🛠️ System Configurations")

            # Simulated Webcam input (Invisible stream orchestrator)
            webcam_input = gr.Image(
                sources=["webcam"],
                type="numpy",
                streaming=True,
                label="Active Video Buffer",
                show_label=True
            )

            # Interactive CVD selection dropdown
            cvd_type = gr.Dropdown(
                choices=["Deuteranopia (Green-Blind)"],
                value="Deuteranopia (Green-Blind)",
                label="CVD Classification Profile"
            )

            # Fine-tuning slider for the math threshold parameter
            threshold_slider = gr.Slider(
                minimum=5,
                maximum=60,
                value=25,
                step=1,
                label="Delta-E Sensitivity Threshold"
            )

            gr.Markdown(
                """> **Presentation Tip:** Use a lower threshold value in low-light environments to mitigate sensor noise in the confusion matrix calculations."""
            )

        # Unified 3-Pane Diagnostic Output Grid
        with gr.Column(scale=3):
            gr.Markdown("### 📊 Real-Time Pipeline Displays")

            with gr.Row():
                out_orig = gr.Image(label="1. Original Live Stream", interactive=False)
                out_sim = gr.Image(label="2. Perceptual CVD Simulation", interactive=False)
                out_mask = gr.Image(label="3. Extracted Confusion Mask", interactive=False)

    # Establish the live execution stream loop
    # As the webcam yields frames, this event handler executes instantly and populates the output grid
    webcam_input.stream(
        fn=process_live_stream,
        inputs=[webcam_input, threshold_slider],
        outputs=[out_orig, out_sim, out_mask],
        queue=True
    )

# Launch the interactive local development server
if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(),share=False)