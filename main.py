#Deuteranopia Simulation
import cv2
import numpy as np

# Deuteranopia matrix (Machado model)
M = np.array([
    [0.430, 0.720, -0.150],
    [0.340, 0.620,  0.040],
    [-0.020, 0.030,  0.990]
], dtype=np.float32)

def simulate_deuteranopia(frame):
    # convert BGR → RGB
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # apply transformation
    out = cv2.transform(img, M)

    # clip values
    out = np.clip(out, 0, 1)

    # back to 8-bit
    out = (out * 255).astype(np.uint8)

    # RGB → BGR
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


# LIVE CAMERA TEST
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    simulated = simulate_deuteranopia(frame)

    cv2.imshow("Original", frame)
    cv2.imshow("Deuteranopia Simulation", simulated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ----------------------------
# 2. Confusion Mask
# ----------------------------
def get_confusion_mask(original, simulated, threshold=25):
    orig_lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB)
    sim_lab = cv2.cvtColor(simulated, cv2.COLOR_BGR2LAB)

    # chromatic channels only
    orig_a, orig_b = orig_lab[:, :, 1], orig_lab[:, :, 2]
    sim_a, sim_b = sim_lab[:, :, 1], sim_lab[:, :, 2]

    diff_a = cv2.absdiff(orig_a, sim_a)
    diff_b = cv2.absdiff(orig_b, sim_b)

    diff = cv2.addWeighted(diff_a, 0.5, diff_b, 0.5, 0)

    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)

    return mask


# ----------------------------
# 3. Run on Image & Process Pipeline
# ----------------------------
img = cv2.imread("test.jpg")

if img is None:
    print("Image not found!")
    exit()

sim = simulate_deuteranopia(img)
mask = get_confusion_mask(img, sim)

# التعديل الجوهري: تحويل القناع إلى 3 قنوات ليطابق أبعاد الصور الأخرى قبل الدمج
mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

# إضافة نصوص توضيحية مباشرة فوق الصور لتمييزها في العرض
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, '1. Original Feed', (20, 40), font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(sim, '2. Simulated Deuteranopia', (20, 40), font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(mask_3ch, '3. Extracted Confusion Mask', (20, 40), font, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

# ----------------------------
# 4. Horizontal Concatenation & Scaling
# ----------------------------
# دمج الصور الثلاث أفقياً في نافذة واحدة
combined_canvas = np.hstack((img, sim, mask_3ch))

# تغيير حجم اللوحة المدمجة لتناسب الشاشة أثناء العرض
scale = 0.45
width = int(combined_canvas.shape[1] * scale)
height = int(combined_canvas.shape[0] * scale)
scaled_presentation = cv2.resize(combined_canvas, (width, height), interpolation=cv2.INTER_AREA)

# عرض النتيجة النهائية الموحدة
cv2.namedWindow("ChromaSight AI - Diagnostic Dashboard", cv2.WINDOW_AUTOSIZE)
cv2.imshow("ChromaSight AI - Diagnostic Dashboard", scaled_presentation)
cv2.waitKey(0)
cv2.destroyAllWindows()