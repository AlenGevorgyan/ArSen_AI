# ArSen_AI

Real-time translator from Armenian Sign Language (ArSL) to text. It uses a normal webcam and runs on a CPU.

<!-- TODO (Alen): record a short demo, save it as docs/demo.gif, and replace the line below with: ![Demo](docs/demo.gif) -->
> Demo GIF coming soon.

## Why

I read a study by Armenian researchers. It says many deaf people in Armenia have trouble in public places, because almost nobody around them understands sign language. Armenia has very few sign language interpreters. Some deaf people are elderly and live alone.

I could not find another program that does this for Armenian Sign Language. This project is a first step.

## What it does

- Opens the webcam and finds your hand.
- Reads the last 30 frames of hand movement.
- Shows the predicted Armenian word on screen, with a confidence score.

The trained model in this repo (`action.h5`) knows 25 single words, for example ողջույն (hello), ցտեսություն (goodbye), ներողություն (sorry), ես (I), դու (you), ինչ (what), որտեղ (where).

## How it works

1. **Hand landmarks.** MediaPipe Hands finds 21 points on the hand in each frame. Each point has x, y, z coordinates.
2. **Sequence.** 30 frames are stacked into one input of shape 30 x 21 x 3 (stored as 30 x 63).
3. **Model.** Three LSTM layers (128, 128, 64 units) with batch normalization and dropout, then dense layers and a softmax over the word classes.
4. **Stable output.** The app predicts on every frame once it has 30 frames. It shows a word only when the same word wins in at least 4 of the last 5 predictions. If the hand leaves the frame, the buffer resets.

Training data is small, so `preprocessing.py` makes 5 versions of every video: the original, a small rotation, a small shift, a small scale change, and added noise.

## Results

The saved model scored 97.4% accuracy on its test split (`model_info.npy`). The test split is a random 15% of the augmented samples. Augmented copies of the same video can end up in both training and test, so this number is higher than real-world accuracy. Live use with a webcam is less accurate, especially in poor light.

## Files

| File | What it does |
|---|---|
| `preprocessing.py` | Reads videos from `videos/<word>/`, extracts landmarks, pads or trims to 30 frames, augments, saves `processed_data/*.npy` |
| `train_model.py` | Trains the LSTM model, saves `action.h5` and `model_info.npy` (labels and test scores), plots training curves |
| `main.py` | Runs the real-time webcam translator |
| `action.h5` | Trained model |
| `model_info.npy` | Label map for the trained model |
| `arial.ttf` | Font used to draw Armenian letters on the video |

## How to run

Python 3.9–3.11 is recommended (TensorFlow 2.13).

```bash
git clone https://github.com/AlenGevorgyan/ArSen_AI.git
cd ArSen_AI
pip install -r requirements.txt
pip install pillow matplotlib
python main.py
```

Press `Q` to quit and `C` to clear the buffer. Hold a sign until the word appears. Hide your hand to start the next sign.

Note: `main.py` looks for `Arial.ttf`, but the file in the repo is `arial.ttf`. On Linux and macOS, rename the file or change `FONT_PATH` in `main.py`.

### Train on your own videos

1. Put videos in `videos/<word>/` (one folder per word, `.mp4`, `.avi` or `.mov`).
2. Run `python preprocessing.py`.
3. In `train_model.py`, set `DATA_PATH = "processed_data"` (it currently points to a Kaggle path).
4. Run `python train_model.py`.

## Limits

- Single words only. It does not build sentences.
- Needs good light. In the dark it guesses random words.
- Uses only the first hand it detects, so two-handed signs are not fully read.
- 25 words, trained on a small set of videos.
- Not tested with deaf users yet.

## Next steps

- Full sentences.
- Work in any light.
- More signs.
- Tests with deaf people in Armenia.
