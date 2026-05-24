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

cap.release()
cv2.destroyAllWindows()