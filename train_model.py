import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

DATA_DIR = 'MP_Data'
sequence_length = 30 # 30 frames per sequence

def main():
    if not os.path.exists(DATA_DIR):
        print(f"Error: Directory '{DATA_DIR}' not found. Please run collect_data.py first.")
        return
        
    # Get list of actions/words from directory names
    actions = np.array([name for name in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, name))])
    
    if len(actions) == 0:
        print("Error: No data sequences found. Please run collect_data.py to gather data.")
        return
        
    print(f"Discovered words: {actions}")
    
    # Create action map to integers
    label_map = {label:num for num, label in enumerate(actions)}
    
    sequences, labels = [], []
    for action in actions:
        action_path = os.path.join(DATA_DIR, action)
        sample_folders = [f for f in os.listdir(action_path) if os.path.isdir(os.path.join(action_path, f))]
        
        for sequence_id in sample_folders:
            window = []
            seq_path = os.path.join(action_path, sequence_id)
            
            # Load 30 frames
            for frame_num in range(sequence_length):
                res_path = os.path.join(seq_path, f"{frame_num}.npy")
                if os.path.exists(res_path):
                    res = np.load(res_path)
                    window.append(res)
                else:
                    break
                    
            if len(window) == sequence_length:
                sequences.append(window)
                labels.append(label_map[action])
                
    if not sequences:
        print("Error: Could not load full sequences. Ensure recorded sequences have 30 frames.")
        return
        
    X = np.array(sequences)
    y = to_categorical(labels).astype(int)
    
    print(f"Data shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    
    # Save the labels mapping to be used in main.py
    np.save('labels.npy', actions)
    
    print("=== Training LSTM Model ===")
    
    # Build Model
    from tensorflow.keras.layers import Dropout
    model = Sequential()
    # Input shape: (sequence_length=30, features=258)
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(sequence_length, 258)))
    model.add(Dropout(0.2))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(Dropout(0.2))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax'))
    
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    
    # Train
    model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test))
    
    print("=== Done Training ===")
    model.save('action.h5')
    print("Model saved as 'action.h5'")
    print("Labels mapped and saved as 'labels.npy'")

if __name__ == '__main__':
    main()
