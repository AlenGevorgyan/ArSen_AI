"""
Part 3: Real-Time Sign Language Recognition
This script is modified to display Armenian Unicode text using Pillow (PIL).
"""

import cv2
import numpy as np
import mediapipe as mp
from tensorflow import keras
from collections import deque, Counter
import time
from PIL import Image, ImageDraw, ImageFont

# ============================================================================
# CONFIGURATION
# ============================================================================
MODEL_PATH = "action.h5"
MODEL_INFO_PATH = "model_info.npy"
CONFIDENCE_THRESHOLD = 0.80
SEQUENCE_LENGTH = 30
BUFFER_SIZE = SEQUENCE_LENGTH

# --- STABILIZATION CONFIG ---
HISTORY_SIZE = 5
CONSISTENCY_THRESHOLD = 0.8
# --------------------------------

# --- ARMENIAN FONT CONFIG ---
# IMPORTANT: You must replace 'path/to/your/ArmenianFont.ttf'
# with the actual path to a Unicode font file on your system.
FONT_PATH = "Arial.ttf"  # Example font file name (must be accessible)
FONT_SIZE_MAIN = 50
FONT_SIZE_CONF = 25
# --------------------------------

# Global font objects
try:
    ARM_FONT_MAIN = ImageFont.truetype(FONT_PATH, FONT_SIZE_MAIN)
    ARM_FONT_CONF = ImageFont.truetype(FONT_PATH, FONT_SIZE_CONF)
    print("✅ Unicode fonts loaded successfully.")
except IOError:
    print(f"❌ Error: Cannot load font file at '{FONT_PATH}'.")
    print("Please ensure the font path is correct and the file exists.")
    exit()

# ============================================================================
# LOAD MODEL AND INFO (UNCHANGED)
# ============================================================================
print("=" * 70)
print("SIGN LANGUAGE TRANSLATOR - REAL-TIME INFERENCE")
print("=" * 70)

print("\n📂 Loading model...")
try:
    model = keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

print("\n📂 Loading model info...")
try:
    model_info = np.load(MODEL_INFO_PATH, allow_pickle=True).item()
    label_map = model_info['label_map']
    # Create inverse mapping (index -> letter)
    inv_label_map = {v: k for k, v in label_map.items()}
    # --- EXAMPLE: MAPPING ENGLISH LABELS TO ARMENIAN WORDS ---
    # You MUST replace the English labels in your 'label_map' values
    # with the Armenian words you want to display.
    # Example for demonstration:
    # inv_label_map[0] = "ԲԱՐԵՎ"  # BAREV (Hello)
    # inv_label_map[1] = "ՇՆՈՐՀԱԿԱԼՈՒԹՅՈՒՆ" # SHNORHAKALUTYUN (Thank You)
    # ---------------------------------------------------------
    print(f"✅ Loaded {len(label_map)} classes")
except Exception as e:
    print(f"❌ Error loading model info: {e}")
    exit()

# ============================================================================
# INITIALIZE MEDIAPIPE (UNCHANGED)
# ============================================================================
print("\n🤖 Initializing MediaPipe...")
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

print("✅ MediaPipe initialized")


# ============================================================================
# NEW: UNICODE DRAWING FUNCTION
# ============================================================================
def draw_text_unicode(image, text, org, font, color):
    """
    Draws Unicode text onto an OpenCV image using PIL.
    """
    # Convert image from OpenCV BGR to PIL RGB
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # Text color must be in RGB tuple format for PIL
    # OpenCV color (B, G, R) needs conversion to PIL color (R, G, B)
    text_color_rgb = (color[2], color[1], color[0])

    draw.text(org, text, font=font, fill=text_color_rgb)

    # Convert back from PIL RGB to OpenCV BGR
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def extract_landmarks(hand_landmarks):
    """
    Extract landmarks from MediaPipe hand detection.
    """
    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.extend([lm.x, lm.y, lm.z])
    return np.array(landmarks)


def draw_styled_landmarks(image, hand_landmarks):
    """
    Draw hand landmarks on image with nice styling.
    """
    mp_drawing.draw_landmarks(
        image,
        hand_landmarks,
        mp_hands.HAND_CONNECTIONS,
        mp_drawing_styles.get_default_hand_landmarks_style(),
        mp_drawing_styles.get_default_hand_connections_style()
    )


def draw_prediction_box(image, text, confidence, color):
    """
    Draw a styled prediction box on the image using Unicode text.
    """

    # Use PIL to measure text size for accurate box drawing
    # Note: PIL's textsize method is deprecated, but simpler for this use case.
    try:
        text_size = ARM_FONT_MAIN.getsize(text)
        conf_text = f"{confidence:.0f}%"
        conf_size = ARM_FONT_CONF.getsize(conf_text)
    except AttributeError:
        # Fallback for newer PIL versions if getsize is removed
        # A more robust solution might require a separate function to calculate text bounds
        print("Warning: Using bounding box for text size calculation.")
        text_bbox = ARM_FONT_MAIN.getbbox(text)
        text_size = (text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1])
        conf_text = f"{confidence:.0f}%"
        conf_bbox = ARM_FONT_CONF.getbbox(conf_text)
        conf_size = (conf_bbox[2] - conf_bbox[0], conf_bbox[3] - conf_bbox[1])
    except Exception as e:
        print(f"Error measuring text size: {e}")
        text_size = (200, 50)  # Safety fallback
        conf_size = (80, 20)

    # Box parameters
    box_width = max(text_size[0], conf_size[0]) + 40
    box_height = text_size[1] + conf_size[1] + 40
    box_x = 20
    box_y = 20

    # Draw semi-transparent box (using OpenCV)
    overlay = image.copy()
    cv2.rectangle(
        overlay,
        (box_x, box_y),
        (box_x + box_width, box_y + box_height),
        (0, 0, 0),
        -1
    )
    cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)

    # Draw border (using OpenCV)
    cv2.rectangle(
        image,
        (box_x, box_y),
        (box_x + box_width, box_y + box_height),
        color,
        3
    )

    # Draw text using the Unicode drawing helper
    text_x = box_x + 20
    text_y = box_y + 15  # Y-origin for PIL is top-left

    image = draw_text_unicode(
        image,
        text,
        (text_x, text_y),
        ARM_FONT_MAIN,
        color
    )

    conf_y = text_y + text_size[1] + 10
    image = draw_text_unicode(
        image,
        conf_text,
        (text_x, conf_y),
        ARM_FONT_CONF,
        (255, 255, 255)  # White color for confidence
    )

    return image


def draw_info_panel(image, fps, buffer_status, stable_status):
    """
    Draw information panel at bottom of screen. Uses OpenCV for simple text
    and rectangles, as complex Unicode isn't needed here.
    """
    height, width, _ = image.shape

    # Create info panel
    panel_height = 60
    overlay = image.copy()
    cv2.rectangle(
        overlay,
        (0, height - panel_height),
        (width, height),
        (0, 0, 0),
        -1
    )
    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)

    # Draw info text
    font = cv2.FONT_HERSHEY_SIMPLEX

    # FPS
    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(
        image,
        fps_text,
        (20, height - 20),
        font,
        0.6,
        (0, 255, 0),
        2
    )

    # Buffer status
    buffer_text = f"Seq Buffer: {buffer_status}/{BUFFER_SIZE}"
    buffer_color = (0, 255, 0) if buffer_status == BUFFER_SIZE else (100, 100, 255)
    cv2.putText(
        image,
        buffer_text,
        (200, height - 20),
        font,
        0.6,
        buffer_color,
        2
    )

    # Stability status
    stability_text = f"Stable: {stable_status}/{HISTORY_SIZE}"
    # Determine color based on whether stability threshold is met
    is_stable = stable_status >= HISTORY_SIZE * CONSISTENCY_THRESHOLD
    stability_color = (0, 255, 0) if is_stable else (255, 100, 0)
    cv2.putText(
        image,
        stability_text,
        (420, height - 20),
        font,
        0.6,
        stability_color,
        2
    )

    # Instructions
    instructions = "Press 'Q' to quit | Press 'C' to clear buffer"
    cv2.putText(
        image,
        instructions,
        (width - 520, height - 20),
        font,
        0.5,
        (200, 200, 200),
        1
    )


# ============================================================================
# MAIN INFERENCE LOOP (draw_prediction_box CALL UPDATED)
# ============================================================================
def run_inference():
    """
    Main function for real-time sign language recognition.
    """
    print("\n" + "=" * 70)
    print("🎥 STARTING WEBCAM INFERENCE")
    print("=" * 70)
    print("\n📹 Opening webcam...")
    print("   • Show your hand gesture and hold until prediction appears.")
    print("   • HIDE HAND to reset for the next sign.")
    print("   • Press 'Q' to quit")
    print("   • Press 'C' to clear buffer\n")

    # Initialize webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Error: Could not open webcam")
        return

    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Initialize buffer for sequence (30 frames of landmarks)
    landmark_buffer = deque(maxlen=BUFFER_SIZE)

    # Initialize buffer for prediction history (5 frames of predicted class index)
    prediction_history = deque(maxlen=HISTORY_SIZE)
    current_stable_prediction_index = -1

    # Variables for display
    current_prediction_label = "Waiting..."
    current_confidence = 0.0
    prediction_color = (255, 255, 255)
    stability_count = 0  # For displaying stability status

    # FPS calculation
    fps = 0
    frame_count = 0
    start_time = time.time()

    print("✅ Webcam started successfully\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)

        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process with MediaPipe
        results = hands.process(frame_rgb)

        # --- State Variables for Frame ---
        is_hand_detected = False
        prediction_made_this_frame = False

        # Check if hand is detected
        if results.multi_hand_landmarks:
            is_hand_detected = True

            # --- DRAW ALL HANDS, PROCESS ONLY THE FIRST ONE ---
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Draw landmarks on frame for ALL detected hands
                draw_styled_landmarks(frame, hand_landmarks)

                # Prediction logic only runs on the first detected hand (i=0)
                if i == 0:
                    # Extract landmark coordinates and append to buffer
                    landmarks = extract_landmarks(hand_landmarks)
                    landmark_buffer.append(landmarks)

                    # --- PREDICTION LOGIC: Predict on every frame once buffer is full ---
                    if len(landmark_buffer) == BUFFER_SIZE:
                        prediction_made_this_frame = True

                        # Prepare sequence for model
                        sequence = np.array(list(landmark_buffer))
                        sequence = np.expand_dims(sequence, axis=0)  # Add batch dimension

                        # Get prediction
                        prediction = model.predict(sequence, verbose=0)[0]
                        predicted_class_index = np.argmax(prediction)
                        confidence = prediction[predicted_class_index]

                        # Append prediction index to history
                        prediction_history.append(predicted_class_index)

                        # --- TEMPORAL FILTERING AND STABILIZATION ---
                        if len(prediction_history) == HISTORY_SIZE:
                            # Find the most common prediction in the last 5 frames
                            counts = Counter(prediction_history)
                            most_common_index, frequency = counts.most_common(1)[0]
                            stability_count = frequency  # Update for display

                            # Check if the most common prediction meets the stability threshold
                            if frequency / HISTORY_SIZE >= CONSISTENCY_THRESHOLD:
                                # Stable prediction found!
                                current_stable_prediction_index = most_common_index

                                # Update display information
                                if confidence >= CONFIDENCE_THRESHOLD:
                                    # Get Armenian label
                                    current_prediction_label = inv_label_map.get(current_stable_prediction_index,
                                                                                 "Unknown")
                                    current_confidence = confidence * 100
                                    prediction_color = (0, 255, 0)  # Green for stable, high confidence
                                else:
                                    # Display the stable class, but warn of low confidence
                                    current_prediction_label = inv_label_map.get(current_stable_prediction_index,
                                                                                 "Unknown")
                                    current_confidence = confidence * 100
                                    prediction_color = (0, 165, 255)  # Orange for stable, low confidence

                            else:
                                # Not stable yet or moving, use a non-Armenian label for status
                                current_prediction_label = "Processing..."
                                current_confidence = confidence * 100
                                prediction_color = (255, 255, 0)  # Yellow for processing

        # --- HAND NOT DETECTED (RESET LOGIC) ---
        if not is_hand_detected:
            # Only clear the buffer if it was previously filled, this forces a reset
            if len(landmark_buffer) > 0:
                landmark_buffer.clear()
                prediction_history.clear()
                current_stable_prediction_index = -1
                stability_count = 0

            current_prediction_label = "No Hand Detected - RESET"  # Non-Armenian status
            current_confidence = 0
            prediction_color = (0, 0, 255)  # Red

        # --- DISPLAY CURRENT STABLE PREDICTION ---
        if not prediction_made_this_frame and current_stable_prediction_index != -1:
            # If no new prediction was made, but a stable one exists, display it
            pass  # Keep the previous display state

        # Calculate FPS
        frame_count += 1
        if frame_count % 10 == 0:
            end_time = time.time()
            fps = 10 / (end_time - start_time)
            start_time = time.time()

        # Draw prediction box (uses the updated Unicode function)
        frame = draw_prediction_box(frame, current_prediction_label, current_confidence, prediction_color)

        # Draw info panel (uses standard OpenCV)
        draw_info_panel(frame, fps, len(landmark_buffer), stability_count)

        # Display frame
        cv2.imshow('Sign Language Translator - Real-Time', frame)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            print("\n👋 Quitting...")
            break
        elif key == ord('c') or key == ord('C'):
            landmark_buffer.clear()
            prediction_history.clear()
            current_stable_prediction_index = -1
            stability_count = 0
            current_prediction_label = "Buffer Cleared - RESET"
            current_confidence = 0
            print("🗑️  Buffer cleared")

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    hands.close()

    print("\n✅ Webcam closed")
    print("=" * 70)
    print("Thank you for using Sign Language Translator!")
    print("=" * 70 + "\n")


# ============================================================================
# RUN
# ============================================================================
if __name__ == "__main__":
    try:
        run_inference()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()