import cv2
import mediapipe as mp
import numpy as np
import os
import time

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

DATA_DIR = 'MP_Data'
no_sequences = 30 # Number of videos/sequences per word
sequence_length = 30 # 30 frames per video

def extract_keypoints(results):
    # Extract pose
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    # Extract left hand
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    # Extract right hand
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    
    # Total features: 132 + 63 + 63 = 258
    return np.concatenate([pose, lh, rh])

def draw_styled_landmarks(image, results):
    # Draw pose connections
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
        mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
    ) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
        mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
        mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
    ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
    )

def main():
    print("=== Sign Language Sequence Data Collection ===")
    word = input("Enter the word/sign you want to record (e.g., 'HELLO', 'STOP'): ").strip().upper()
    if not word:
        print("Label cannot be empty. Exiting.")
        return
        
    # Create directory for the word if it doesn't exist
    word_path = os.path.join(DATA_DIR, word)
    if not os.path.exists(word_path):
        os.makedirs(word_path)
    
    # Check existing sequences to continue
    existing_seqs = [int(f) for f in os.listdir(word_path) if os.path.isdir(os.path.join(word_path, f))]
    start_seq = max(existing_seqs) + 1 if existing_seqs else 0
    
    try:
        num_to_record = int(input(f"How many sequences do you want to record? (Default {no_sequences}): ") or no_sequences)
    except ValueError:
        print("Invalid number. Exiting.")
        return
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
        
    print("\nPress 'Q' to quit anytime.")
    print(f"Get ready to record '{word}'.")
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for seq in range(start_seq, start_seq + num_to_record):
            # Create sequence directory
            seq_path = os.path.join(word_path, str(seq))
            if not os.path.exists(seq_path):
                os.makedirs(seq_path)
                
            print(f"\nRecording sequence {seq} in 3 seconds...")
            
            # Draw countdown before each sequence
            for i in range(3, 0, -1):
                success, image = cap.read()
                if success:
                    image = cv2.flip(image, 1) # Mirror selfie view
                    cv2.putText(image, f"STARTING IN {i}", (120,200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255, 0), 4, cv2.LINE_AA)
                    cv2.imshow('MediaPipe Holistic Record', image)
                    cv2.waitKey(1000)
                else:
                    time.sleep(1)
                    
            # Record `sequence_length` frames
            for frame_num in range(sequence_length):
                success, image = cap.read()
                if not success:
                    image = np.zeros((480, 640, 3), dtype=np.uint8)

                image = cv2.flip(image, 1) # Mirror selfie view
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_rgb.flags.writeable = False
                results = holistic.process(image_rgb)

                # Draw annotations
                image_rgb.flags.writeable = True
                image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                
                draw_styled_landmarks(image, results)
                
                # Show recording status
                cv2.putText(image, f"Word: '{word}' | Seq: {seq}/{start_seq + num_to_record - 1} | Frame: {frame_num}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow('MediaPipe Holistic Record', image)
                
                # Extract and save keypoints
                keypoints = extract_keypoints(results)
                if len(keypoints) != 258:
                    print("Bad keypoints detected, fixing...")
                    keypoints = np.zeros(258)
                npy_path = os.path.join(seq_path, f"{frame_num}.npy")
                np.save(npy_path, keypoints.astype(np.float32))
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nRecording stopped by user.")
                    cap.release()
                    cv2.destroyAllWindows()
                    return
                    
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nSuccessfully recorded sequences for '{word}'.")

if __name__ == '__main__':
    main()
