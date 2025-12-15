"""
live_demo.py

Run-time drowsiness detection demo using Haar cascades + a saved Keras model (.h5).

Usage:
  python live_demo.py --model_path mobilenet_model.h5
  python live_demo.py --model_path custom_cnn.h5 --threshold 0.5 --consec 15 --camera 0

Requirements:
  pip install tensorflow opencv-python numpy simpleaudio tqdm

If simpleaudio isn't installed, the script still runs but will not play sound (it will print instead).
"""
import argparse
import time
import threading
import sys

import numpy as np
import cv2

import tensorflow as tf
from tensorflow.keras.models import load_model

# Optional sound: simpleaudio (pip install simpleaudio)
try:
    import simpleaudio as sa
    SIMPLEAUDIO_AVAILABLE = True
except Exception:
    SIMPLEAUDIO_AVAILABLE = False

# -------------------------
# Alarm sound generator
# -------------------------
def make_beep(duration_s=0.4, freq=880.0, sample_rate=44100, volume=0.3):
    """
    Create a numpy array with a sine wave beep.
    Returns int16 array suitable for simpleaudio.
    """
    t = np.linspace(0, duration_s, int(sample_rate*duration_s), False)
    tone = np.sin(freq * 2 * np.pi * t)
    audio = tone * (32767 * volume)
    audio = audio.astype(np.int16)
    return audio

_beep_audio = make_beep()

def play_beep_nonblocking():
    if SIMPLEAUDIO_AVAILABLE:
        def _play():
            try:
                sa.play_buffer(_beep_audio, 1, 2, 44100)
            except Exception:
                pass
        thr = threading.Thread(target=_play, daemon=True)
        thr.start()
    else:
        # Fallback: try system beep (Windows), otherwise print
        try:
            if sys.platform.startswith('win'):
                import winsound
                winsound.Beep(880, 400)
            else:
                # ASCII BEL (may not work in modern terminals)
                print('\a', end='', flush=True)
        except Exception:
            print("[ALARM] DROWSY!")

# -------------------------
# Helper: infer model input details
# -------------------------
def get_model_input_info(model):
    """
    Returns (input_size, channels, expects_grayscale_bool)
    Accepts typical shapes: (None, H, W, C) or (None, H, W) etc.
    """
    shape = model.input_shape
    # shape can be tuple or list (for multiple inputs); handle common single-input case
    if isinstance(shape, list):
        shape = shape[0]
    
    # Convert to list to handle TensorShape objects
    shape = list(shape)
    
    # shape like [None, H, W, C] or [None, H, W]
    if len(shape) == 4:
        _, h, w, c = shape
    elif len(shape) == 3:
        _, h, w = shape
        c = 1
    else:
        raise ValueError(f"Unsupported model input shape: {shape}")
    if h is None:
        # try to infer from w if present
        raise ValueError("Model has undefined input height. Use a model with static input size.")
    expects_grayscale = (c == 1)
    return (int(h), int(w), int(c), expects_grayscale)

# -------------------------
# Preprocessing helper
# -------------------------
def preprocess_eye(eye_img, target_h, target_w, expects_grayscale):
    """
    eye_img: BGR image crop from OpenCV
    returns: numpy batch [1, h, w, c] normalized float32
    """
    if expects_grayscale:
        eye = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
        eye = cv2.resize(eye, (target_w, target_h))
        eye = eye.astype('float32') / 255.0
        eye = np.expand_dims(eye, axis=(0, -1))  # [1,h,w,1]
    else:
        # convert BGR->RGB
        eye = cv2.cvtColor(eye_img, cv2.COLOR_BGR2RGB)
        eye = cv2.resize(eye, (target_w, target_h))
        eye = eye.astype('float32') / 255.0
        eye = np.expand_dims(eye, axis=0)  # [1,h,w,3]
    return eye

# -------------------------
# Main live loop
# -------------------------
def live_demo(model_path,
              camera_index=0,
              threshold=0.5,
              consecutive_frames=15,
              face_scale=1.1,
              eye_scale=1.1,
              show_fps=True,
              draw_boxes=True):
    print("Loading model:", model_path)
    model = load_model(model_path)
    h, w, c, expects_grayscale = get_model_input_info(model)
    print(f"Model input -> H: {h}, W: {w}, C: {c}, expects_grayscale: {expects_grayscale}")

    # Haar cascades bundled with OpenCV
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Unable to open camera.")

    closed_count = 0
    alarm_on = False
    last_time = time.time()
    fps = 0.0

    print("Starting webcam. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        t0 = time.time()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # faces: scaleFactor=face_scale, minNeighbors default 5; tweak if many false positives
        faces = face_cascade.detectMultiScale(gray, scaleFactor=face_scale, minNeighbors=5)

        frame_label = "AWAKE"
        frame_color = (0, 255, 0)  # green

        # If no face detected, assume we cannot classify — increment closed_count slowly
        if len(faces) == 0:
            closed_count = min(closed_count + 1, consecutive_frames + 5)
        else:
            any_eye_detected = False
            eye_probs = []
            # for each face, detect eyes inside face ROI
            for (fx, fy, fw, fh) in faces:
                roi_gray = gray[fy:fy+fh, fx:fx+fw]
                roi_color = frame[fy:fy+fh, fx:fx+fw]
                eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=eye_scale, minNeighbors=5, minSize=(20, 20))
                if len(eyes) == 0:
                    # No eyes detected for this face — might be closed
                    continue
                any_eye_detected = True
                for (ex, ey, ew, eh) in eyes:
                    # extract eye region with some padding
                    pad_x = int(0.1 * ew)
                    pad_y = int(0.1 * eh)
                    sx = max(ex - pad_x, 0)
                    sy = max(ey - pad_y, 0)
                    ex2 = min(ex + ew + pad_x, fw)
                    ey2 = min(ey + eh + pad_y, fh)
                    eye_img = roi_color[sy:ey2, sx:ex2]
                    if eye_img.size == 0:
                        continue
                    try:
                        inp = preprocess_eye(eye_img, h, w, expects_grayscale)
                        p = float(model.predict(inp, verbose=0)[0][0])  # probability of "open" (we trained with open=1)
                        eye_probs.append(p)
                        # draw each eye box if desired
                        if draw_boxes:
                            # map eye coords relative to original frame
                            ex1 = fx + sx
                            ey1 = fy + sy
                            ex2b = fx + ex2
                            ey2b = fy + ey2
                            cv2.rectangle(frame, (ex1, ey1), (ex2b, ey2b), (255, 200, 0), 1)
                            cv2.putText(frame, f"{p:.2f}", (ex1, ey1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,200,0), 1)
                    except Exception as e:
                        # any failure in preprocessing/predict will be ignored for that eye
                        # print("Predict error:", e)
                        continue

            # decide based on mean probability across all detected eyes across all faces
            if len(eye_probs) == 0:
                # no eyes detected for faces -> likely closed
                closed_count += 1
            else:
                mean_prob_open = np.mean(eye_probs)
                # if mean_prob_open > threshold -> eyes open
                if mean_prob_open > threshold:
                    # awake: decay count
                    closed_count = max(0, closed_count - 1)
                else:
                    closed_count += 1

        # Decide drowsiness
        if closed_count >= consecutive_frames:
            frame_label = "DROWSY"
            frame_color = (0, 0, 255)  # red
            if not alarm_on:
                alarm_on = True
                # Play alarm (non-blocking)
                play_beep_nonblocking()
        else:
            frame_label = "AWAKE"
            frame_color = (0, 255, 0)
            alarm_on = False

        # Draw face boxes and label
        for (fx, fy, fw, fh) in faces:
            cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), frame_color, 2)

        # status panel
        cv2.putText(frame, f"State: {frame_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, frame_color, 2)
        cv2.putText(frame, f"ClosedCount: {closed_count}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        # FPS computation & display
        if show_fps:
            t1 = time.time()
            fps = 0.9*fps + 0.1*(1.0/(t1 - last_time + 1e-8))
            last_time = t1
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

        cv2.imshow("Drowsiness Detection (press 'q' to quit)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# -------------------------
# CLI
# -------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Live drowsiness demo (Haar cascades + Keras model)")
    p.add_argument('--model_path', required=True, help='Path to Keras model (.h5)')
    p.add_argument('--camera', type=int, default=0, help='Camera index (default 0)')
    p.add_argument('--threshold', type=float, default=0.5, help='Probability threshold for OPEN (default 0.5)')
    p.add_argument('--consec', type=int, default=15, help='Consecutive closed frames to trigger alarm (default 15)')
    p.add_argument('--face_scale', type=float, default=1.1, help='Haar face scaleFactor (default 1.1)')
    p.add_argument('--eye_scale', type=float, default=1.1, help='Haar eye scaleFactor (default 1.1)')
    p.add_argument('--no_fps', action='store_true', help='Disable FPS display')
    p.add_argument('--no_boxes', action='store_true', help='Disable drawing of detected eyes/face boxes')
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    try:
        live_demo(model_path=args.model_path,
                  camera_index=args.camera,
                  threshold=args.threshold,
                  consecutive_frames=args.consec,
                  face_scale=args.face_scale,
                  eye_scale=args.eye_scale,
                  show_fps=(not args.no_fps),
                  draw_boxes=(not args.no_boxes))
    except Exception as e:
        import traceback
        print("Error:", e)
        traceback.print_exc()
        sys.exit(1)
