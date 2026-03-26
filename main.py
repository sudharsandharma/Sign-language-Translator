import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import os

MODEL_FILE = 'action.h5'
LABELS_FILE = 'labels.npy'
SEQUENCE_LENGTH = 30

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])

def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
        mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
    ) 
    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
        mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
        mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
    ) 
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
    )

def main():
    if not os.path.exists(MODEL_FILE) or not os.path.exists(LABELS_FILE):
        print(f"Error: {MODEL_FILE} or {LABELS_FILE} not found. Please run train_model.py first.")
        return

    # Load model and labels
    model = load_model(MODEL_FILE)
    actions = np.load(LABELS_FILE)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
        
    print("=== Sign Language Real-Time Translator ===")
    print("Press 'Q' to quit.")

    sequence = []
    current_word = ''
    predictions = []
    threshold = 0.8
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            image = cv2.flip(image, 1) # Selfie view
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = holistic.process(image_rgb)
            
            image_rgb.flags.writeable = True
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            # Draw landmarks
            draw_styled_landmarks(image_bgr, results)

            # Extract features and handle sequence
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-SEQUENCE_LENGTH:] # Keep last 30 frames
            
            if len(sequence) == SEQUENCE_LENGTH:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                predicted_class = np.argmax(res)
                predictions.append(predicted_class)
                
                # Stabilization (only output if prediction is consistent)
                if len(predictions) > 10 and len(set(predictions[-10:])) == 1:
                    # If confidence is above threshold
                    if res[predicted_class] > threshold:
                        current_word = actions[predicted_class]
                
                # Predict text overlay
                cv2.rectangle(image_bgr, (0,0), (640, 40), (245, 117, 16), -1)
                cv2.putText(image_bgr, f'Translated: {current_word}', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image_bgr, f'({res[predicted_class]*100:.1f}%)', (480,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                cv2.putText(image_bgr, f'Buffering... {len(sequence)}/30', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.imshow('Sign Language Translator', image_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
