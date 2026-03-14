import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

def main():
    # Initialize camera
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to grab frame")
            break
        # Flip frame
        frame = cv2.flip(frame, 1)
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Process the frame to find hands
        result = hands.process(rgb_frame)
        # If a hand is detected, process the landmarks
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Draw the basic web
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                # Get the height and width of camera frame
                h, w, c = frame.shape
                # Isolate Landmark 8 (Index Finger Tip)
                index_finger = hand_landmarks.landmark[8]
                # Convert the normalized ratio (0.0 - 1.0) into pixel coordinates
                cx, cy = int(index_finger.x * w), int(index_finger.y * h)
                cv2.circle(frame, (cx, cy), 15, (255, 0, 0), cv2.FILLED)
                print(f"Index Finger Tip - X: {cx}, Y: {cy}")
        cv2.imshow("Vision Mouse Control", frame)
        # Press esc to quit
        if cv2.waitKey(1) & 0xFF == 27:
            break

if __name__ == "__main__":
    main()