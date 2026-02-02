import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

# ================= LOAD MODEL =================
model = tf.keras.models.load_model("Ayush_digit_model.h5")

# ================= MEDIAPIPE =================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ================= SIZES =================
CAM_W, CAM_H = 640, 480
CANVAS_W, CANVAS_H = 640, 480

canvas = np.zeros((CANVAS_H, CANVAS_W), dtype=np.uint8)

# ================= CAMERA =================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

prev_x, prev_y = None, None

# ================= MNIST PREPROCESS =================
def preprocess(img):
    _, thresh = cv2.threshold(img, 20, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)

    if coords is None:
        return None

    x, y, w, h = cv2.boundingRect(coords)
    digit = thresh[y:y+h, x:x+w]

    size = max(w, h)
    square = np.zeros((size, size), dtype=np.uint8)
    square[
        (size - h)//2:(size - h)//2 + h,
        (size - w)//2:(size - w)//2 + w
    ] = digit

    digit = cv2.resize(square, (28, 28))
    digit = digit.astype("float32") / 255.0
    digit = digit.reshape(1, 28, 28, 1)
    return digit

# ================= PREDICT =================
def predict_digit(img):
    processed = preprocess(img)
    if processed is None:
        return "-", 0.0

    preds = model.predict(processed, verbose=0)[0]
    digit = np.argmax(preds)
    confidence = preds[digit]
    return digit, confidence

# ================= CONFIDENCE BAR =================
def draw_confidence_bar(frame, confidence):
    bar_x, bar_y = CAM_W + 500, 20
    bar_w, bar_h = 20, 200

    # Background
    cv2.rectangle(frame, (bar_x, bar_y),
                  (bar_x + bar_w, bar_y + bar_h),
                  (50, 50, 50), 2)

    filled = int(bar_h * confidence)

    # Color gradient (red → green)
    r = int(255 * (1 - confidence))
    g = int(255 * confidence)
    color = (0, g, r)

    cv2.rectangle(
        frame,
        (bar_x, bar_y + bar_h - filled),
        (bar_x + bar_w, bar_y + bar_h),
        color,
        -1
    )

    cv2.putText(
        frame,
        f"{int(confidence * 100)}%",
        (bar_x - 10, bar_y + bar_h + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

# ================= MAIN LOOP =================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        lm = hand.landmark

        ix, iy = int(lm[8].x * w), int(lm[8].y * h)

        fingers = [
            lm[8].y < lm[6].y,
            lm[12].y < lm[10].y,
            lm[16].y < lm[14].y,
            lm[20].y < lm[18].y
        ]
        count = sum(fingers)

        cx = int(ix * CANVAS_W / w)
        cy = int(iy * CANVAS_H / h)

        if count == 1:  # Draw
            if prev_x is not None:
                cv2.line(canvas, (prev_x, prev_y), (cx, cy), 255, 14)
            prev_x, prev_y = cx, cy

        elif count >= 4:  # Erase
            cv2.circle(canvas, (cx, cy), 40, 0, -1)
            prev_x, prev_y = None, None
        else:
            prev_x, prev_y = None, None

    digit, confidence = predict_digit(canvas)

    canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    combined = np.hstack([frame, canvas_bgr])

    cv2.putText(
        combined,
        f"Prediction: {digit}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (0, 0, 255),
        3
    )

    draw_confidence_bar(combined, confidence)

    cv2.putText(
        combined,
        "Index = Draw | Palm = Erase | C = Clear | ESC = Exit",
        (20, CAM_H - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.imshow("Air Canvas Digit Recognition", combined)

    key = cv2.waitKey(1)
    if key == ord('c'):
        canvas[:] = 0
    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()
