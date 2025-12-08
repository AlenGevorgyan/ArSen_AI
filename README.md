# 🖐️ Real-Time Sign Language Translator (MediaPipe + LSTM)

A real-time Computer Vision application that translates Sign Language into text using a webcam.

Unlike traditional CNN approaches that analyze raw pixels, this project uses **MediaPipe** to extract hand skeleton landmarks and an **LSTM (Long Short-Term Memory)** neural network to analyze the temporal sequence of movements. This makes the model **lightweight**, **fast**, and capable of learning from small datasets.

## 🚀 Key Features

  * **Real-Time Performance:** Runs smoothly on CPU (no heavy GPU required).
  * **Data Efficient:** Uses skeleton coordinates (x, y, z) instead of images, reducing noise.
  * **Robustness:** Includes data augmentation (rotation/shifting) to handle small datasets (10 videos/class).
  * **Architecture:** MediaPipe Hands → LSTM/GRU Network → Softmax Classification.

-----

## 🛠️ Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/Alen-Gevorgyan/ArSen_AI.git
    cd ArSen_AI
    ```

2.  **Create a virtual environment (Optional but Recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    ```

3.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

### `requirements.txt`

```text
opencv-python
mediapipe
tensorflow
numpy
scikit-learn
matplotlib
```

-----

## 📂 Project Structure

It is critical to organize your video data exactly as shown below before running the scripts.

```text
├── Dataset/                 # Put your raw videos here
│   ├── class_1/
│   │   ├── video1.mp4
│   │   ├── ...
│   ├── class_2/
│   ├── ...
│   └── class_n/
├── preprocessing.py    # Extracts landmarks & augments data
├── train_model.py         # Trains the LSTM model
├── main.py           # Runs the real-time webcam translator
└── README.md
```

-----

## 🏃‍♂️ Usage Instructions

### Step 1: Data Processing & Augmentation

Extracts hand landmarks from your videos. Since the dataset is small (10 videos/class), this script also applies mathematical augmentation (rotation/shifts) to increase the training data by 5x.

```bash
python preprocessing.py
```

  * **Output:** Generates `X.npy` (features) and `y.npy` (labels).

### Step 2: Train the Model

Trains the LSTM network on the extracted coordinate sequences.

```bash
python train_model.py
```

  * **Output:** Saves the trained model weights to `action.h5`.
  * *Note:* Look for validation accuracy \> 85%. If it's lower, consider recording 5 more videos per class.

### Step 3: Real-Time Translation

Launches the webcam, detects hands, and predicts the sign in real-time.

```bash
python main.py
```

  * Press `q` to quit the application.

-----

## 🧠 Model Architecture Details

### Why not CNN?

Using CNNs on raw video frames requires thousands of samples to learn invariance (lighting, background, skin tone). By using **MediaPipe**, we abstract the visual data into **21 points (Landmarks)**.

1.  **Input:** 30 Frames $\times$ 21 Landmarks $\times$ 3 Coordinates (x, y, z).
2.  **LSTM Layers:** Learns the "history" of the movement (sequence).
3.  **Dense Layer:** Classifies the movement into one of 26 categories.

-----

## ⚠️ Troubleshooting

**1. Model is predicting random words?**

  * Ensure you have good lighting. MediaPipe needs to see the hand clearly.
  * Check if the model is overfitting. If Training Accuracy is 99% but Validation is 50%, you need more data augmentation.

**2. "No Hand Detected" errors?**

  * Make sure your hand is visible in the frame before the recording sequence starts.

**3. Laggy performance?**

  * The model is very light. If it lags, it is likely the drawing functions in OpenCV. Try disabling the skeleton drawing in `main.py`.

-----

## 📝 Future Improvements

  * Add a "Space" and "Delete" gesture to form sentences.
  * Implement Text-to-Speech (TTS) to read the predicted words aloud.
  * Deploy as a web app.

-----