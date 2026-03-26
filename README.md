# Custom Sign Language Translator

This project uses Python, OpenCV, MediaPipe, and Scikit-learn to build a real-time, customizable sign language translator.

Because a comprehensive sign language model requires significant training data, this project is built as a pipeline that allows you to train the system on your own signs using your webcam!

## Workflow

### 1. Installation
Install the required dependencies using pip:
```bash
pip install -r requirements.txt
```

### 2. Collect Data
Run the data collection script to record landmarks for different signs you want the model to learn.
```bash
python collect_data.py
```
- It will prompt you for a label (e.g., `A`, `B`, `Hello`).
- It will prompt you for the number of samples (100-200 is recommended per sign).
- The webcam will open, and you must hold your hand in the correct shape for that sign. Keep moving your hand slightly to get a varied dataset!
- Repeat this script for **at least 2 different signs**.

### 3. Train Model
Once you have collected data for multiple signs, run the training script.
```bash
python train_model.py
```
This reads `landmarks_data.csv`, trains a `RandomForestClassifier`, and outputs the accuracy score. Finally, it saves the trained model to `model.pkl`.

### 4. Run the Translator
Run the main script to start translating in real-time!
```bash
python main.py
```
It will open your webcam, detect your hand, extract the features, and predict the sign using the trained model.

## Note
- If you're using this in an IDE (like VSCode or PyCharm), ensure your active directory is set to where these files are located. 
- Ensure you have a working webcam.
