import cv2
import mediapipe as mp
import pyautogui
import time
from evdev import UInput, ecodes as e, AbsInfo

pyautogui.PAUSE = 0 # disable delaey

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
    screen_width, screen_height = pyautogui.size()

    capabilities = {
        e.EV_KEY: [e.BTN_LEFT, e.BTN_RIGHT],
        e.EV_ABS: [
            (e.ABS_X, AbsInfo(value=0, min=0, max=screen_width, fuzz=0, flat=0, resolution=0)),
            (e.ABS_Y, AbsInfo(value=0, min=0, max=screen_height, fuzz=0, flat=0, resolution=0))
        ]
    }
    
    # Create the virtual device
    ui = UInput(capabilities, name='vision-mouse-controller')

    pTime = 0
    smoothening = 7 
    pre_x, pre_y = 0, 0  # Previous location X and Y
    cur_x, cur_y = 0, 0  # Current location X and Y

    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to grab frame")
            break
        
        # Flip frame
        frame = cv2.flip(frame, 1)
        
        # Get the height and width of camera frame
        h, w, c = frame.shape
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame to find hands
        result = hands.process(rgb_frame)
        
        # If a hand is detected, process the landmarks
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Draw the basic web
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                # Isolate Landmark 8 (Index Finger Tip)
                index_finger = hand_landmarks.landmark[8]
                # Convert the normalized ratio (0.0 - 1.0) into pixel coordinates
                cx, cy = int(index_finger.x * w), int(index_finger.y * h)
                # Convert normalized ratios to actual screen coordinates
                screen_x = int(index_finger.x * screen_width)
                screen_y = int(index_finger.y * screen_height)
                
                # Calculate the smoothed coordinates
                cur_x = pre_x + (screen_x - pre_x) / smoothening
                cur_y = pre_y + (screen_y - pre_y) / smoothening
                
                # Move the virtual mouse using evdev
                ui.write(e.EV_ABS, e.ABS_X, int(cur_x))
                ui.write(e.EV_ABS, e.ABS_Y, int(cur_y))
                ui.syn()  # Synchronize the events to apply them immediately
                
                # Update the previous location for the next frame
                pre_x, pre_y = cur_x, cur_y
                
                # Draw the blue circle so you can see where your finger is
                cv2.circle(frame, (cx, cy), 15, (255, 0, 0), cv2.FILLED)
                print(f"Index Finger Tip - X: {cx}, Y: {cy}")
        
        cTime = time.time()
        fps = 1 / (cTime - pTime) if pTime != 0 else 0
        pTime = cTime
        cv2.putText(frame, f'FPS: {int(fps)}', (w - 150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow("Vision Mouse Control", frame)
        # Press esc to quit
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # cleanup
    cap.release()
    cv2.destroyAllWindows()
    ui.close()

if __name__ == "__main__":
    main()