"""
===================================================================
   COMPUTER VISION MODULE FOR ROBOT TELEOPERATION (ALL-IN-ONE)
===================================================================
Features:
 - Config & Tracking (MediaPipe)
 - Smart Gesture Detection & Temporal Smoothing
 - Gesture & Text Inference Engines
 - Command Queue & Cooldown Management
 - Complete UI Dashboard & FPS Performance Monitor
===================================================================
"""

import cv2
import time
import logging
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict, Any
from collections import deque, Counter
from queue import Queue
import mediapipe as mp

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] %(message)s")


# ==========================================
# 1. CONFIGURATION
# ==========================================
@dataclass(frozen=True)
class Config:
    # Camera Settings
    CAM_INDEX: int = 0
    WIDTH: int = 1280
    HEIGHT: int = 720
    
    # Vision & Detection Thresholds
    MAX_HANDS: int = 1
    MIN_DETECTION_CONF: float = 0.7
    MIN_TRACKING_CONF: float = 0.7
    SMOOTHING_WINDOW: int = 5
    
    # Command Engine
    COOLDOWN_SEC: float = 0.8
    QUEUE_SIZE: int = 10
    
    # UI Styling (BGR Format)
    TEXT_COLOR: Tuple[int, int, int] = (0, 255, 0)       # Green
    HIGHLIGHT_COLOR: Tuple[int, int, int] = (0, 165, 255) # Orange
    FONT_SCALE: float = 0.7
    THICKNESS: int = 2


# ==========================================
# 2. HAND TRACKER
# ==========================================
class HandTracker:
    """Handles camera input and landmark extraction using MediaPipe."""
    def __init__(self, config: Config):
        self.config = config
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=self.config.MAX_HANDS,
            min_detection_confidence=self.config.MIN_DETECTION_CONF,
            min_tracking_confidence=self.config.MIN_TRACKING_CONF
        )

    def process_frame(self, frame):
        """Extracts landmark objects, landmark list, and identifies Left/Right hand."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            hand_landmarks_obj = results.multi_hand_landmarks[0]
            landmarks = hand_landmarks_obj.landmark
            handedness = results.multi_handedness[0].classification[0].label
            return hand_landmarks_obj, landmarks, handedness
        return None, None, None

    def draw_landmarks(self, frame, hand_landmarks_obj) -> None:
        """Draws hand connections and joint points."""
        if hand_landmarks_obj:
            self.mp_draw.draw_landmarks(
                frame,
                hand_landmarks_obj,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_styles.get_default_hand_landmarks_style(),
                self.mp_styles.get_default_hand_connections_style()
            )


# ==========================================
# 3. GESTURE DETECTOR
# ==========================================
class GestureDetector:
    """Calculates finger extension states and applies temporal smoothing."""
    def __init__(self, config: Config):
        self.config = config
        self.history: deque = deque(maxlen=self.config.SMOOTHING_WINDOW)

    def _get_finger_states(self, lm, handedness: str) -> List[bool]:
        """Returns boolean array [Thumb, Index, Middle, Ring, Pinky]."""
        fingers = []
        
        # 1. Thumb State (Horizontal comparison based on hand side)
        if handedness == "Right":
            fingers.append(lm[4].x < lm[3].x)
        else:
            fingers.append(lm[4].x > lm[3].x)
            
        # 2. Other 4 Fingers (Vertical comparison: Tip higher than PIP)
        tip_ids = [8, 12, 16, 20]
        for tip in tip_ids:
            fingers.append(lm[tip].y < lm[tip - 2].y)
            
        return fingers

    def detect_raw_gesture(self, lm, handedness: str) -> Tuple[str, int]:
        """Translates finger orientations to gestures."""
        if not lm:
            return "NO_HAND", 0

        states = self._get_finger_states(lm, handedness)
        finger_count = sum(states)

        # Gesture rules
        if states == [True, False, False, False, False]:
            gesture = "THUMBS_UP"
        elif states == [False, True, False, False, False]:
            gesture = "POINTING_FORWARD"
        elif states == [False, True, True, False, False]:
            gesture = "PEACE_TURN"
        elif states == [True, True, False, False, True]:
            gesture = "ROCK_MODE"
        elif finger_count == 5:
            gesture = "STOP_ALL"
        elif finger_count == 0:
            gesture = "FIST_GRAB"
        else:
            gesture = f"CUSTOM_{finger_count}_FINGERS"

        return gesture, finger_count

    def get_smoothed_gesture(self, raw_gesture: str) -> str:
        """Applies majority voting over recent frames to stabilize output."""
        self.history.append(raw_gesture)
        most_common = Counter(self.history).most_common(1)
        return most_common[0][0] if most_common else "UNKNOWN"


# ==========================================
# 4. INFERENCE ENGINES (GESTURE & TEXT)
# ==========================================
class GestureInference:
    """High-level gesture inference module mapping gestures to robot commands."""
    def __init__(self, config: Config):
        self.detector = GestureDetector(config)
        self.action_mapping: Dict[str, Dict[str, Any]] = {
            "THUMBS_UP": {"action": "LIFT_ARM", "speed": 0.5},
            "POINTING_FORWARD": {"action": "MOVE_FORWARD", "speed": 1.0},
            "PEACE_TURN": {"action": "TURN_LEFT", "speed": 0.5},
            "FIST_GRAB": {"action": "GRAB_OBJECT", "speed": 0.0},
            "STOP_ALL": {"action": "EMERGENCY_STOP", "speed": 0.0},
            "NO_HAND": {"action": "IDLE", "speed": 0.0}
        }

    def infer(self, landmarks, handedness: str) -> Dict[str, Any]:
        if not landmarks:
            return {"gesture": "NO_HAND", "action": "IDLE", "finger_count": 0, "speed": 0.0}

        raw_gesture, finger_count = self.detector.detect_raw_gesture(landmarks, handedness)
        smoothed = self.detector.get_smoothed_gesture(raw_gesture)
        mapped = self.action_mapping.get(smoothed, {"action": "CUSTOM_ACTION", "speed": 0.5})

        return {
            "gesture": smoothed,
            "finger_count": finger_count,
            "action": mapped["action"],
            "speed": mapped["speed"]
        }


class TextInference:
    """Inference engine for parsing direct text commands."""
    def __init__(self):
        self.keywords = {
            "forward": "MOVE_FORWARD", "back": "MOVE_BACKWARD",
            "stop": "EMERGENCY_STOP", "left": "TURN_LEFT",
            "right": "TURN_RIGHT", "grab": "GRAB_OBJECT"
        }

    def infer_from_text(self, text: str) -> Dict[str, Any]:
        clean = text.strip().lower()
        action = "UNKNOWN_TEXT_COMMAND"
        for k, v in self.keywords.items():
            if k in clean:
                action = v
                break
        return {"raw_text": text, "action": action}


# ==========================================
# 5. COMMAND MANAGER
# ==========================================
class CommandManager:
    """Manages command dispatch queue, duplicate prevention, and cooldowns."""
    def __init__(self, config: Config):
        self.config = config
        self.queue: Queue = Queue(maxsize=self.config.QUEUE_SIZE)
        self.last_command: Optional[str] = None
        self.last_time: float = 0.0

    def process_command(self, action: str) -> Optional[str]:
        if action in ["IDLE", "UNKNOWN_ACTION"]:
            return None

        now = time.time()
        # Cooldown & Duplicate Check
        if (now - self.last_time) >= self.config.COOLDOWN_SEC:
            if action != self.last_command:
                self.last_command = action
                self.last_time = now
                if not self.queue.full():
                    self.queue.put(action)
                    logging.info(f"[DISPATCH TO PYBULLET] -> Action: {action}")
                    return action
        return None

    def pop_command_for_pybullet(self) -> Optional[str]:
        """Interface method for the PyBullet simulator team."""
        if not self.queue.empty():
            return self.queue.get()
        return None


# ==========================================
# 6. MAIN PIPELINE & UI RUNNER
# ==========================================
def main():
    cfg = Config()
    
    cap = cv2.VideoCapture(cfg.CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.HEIGHT)

    tracker = HandTracker(cfg)
    gesture_engine = GestureInference(cfg)
    cmd_manager = CommandManager(cfg)

    prev_time = 0.0
    print("[INFO] Computer Vision Pipeline Initialized. Press 'q' to Quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read from camera.")
            break

        # Flip for intuitive selfie view
        frame = cv2.flip(frame, 1)

        # FPS Calculation
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0.0
        prev_time = curr_time

        # 1. Vision Processing Pipeline
        hand_landmarks_obj, landmarks, handedness = tracker.process_frame(frame)
        
        inference_res = {"gesture": "NO_HAND", "action": "IDLE", "finger_count": 0}
        
        if landmarks:
            tracker.draw_landmarks(frame, hand_landmarks_obj)
            inference_res = gesture_engine.infer(landmarks, handedness)
            cmd_manager.process_command(inference_res["action"])

        # 2. Render Telemetry UI Dashboard
        cv2.rectangle(frame, (10, 10), (460, 220), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (460, 220), cfg.HIGHLIGHT_COLOR, 2)

        cv2.putText(frame, f"FPS: {int(fps)}", (25, 45), 
                    cv2.FONT_HERSHEY_SIMPLEX, cfg.FONT_SCALE, cfg.TEXT_COLOR, cfg.THICKNESS)
        cv2.putText(frame, f"Hand: {handedness if handedness else 'N/A'}", (25, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, cfg.FONT_SCALE, cfg.TEXT_COLOR, cfg.THICKNESS)
        cv2.putText(frame, f"Fingers Count: {inference_res['finger_count']}", (25, 115), 
                    cv2.FONT_HERSHEY_SIMPLEX, cfg.FONT_SCALE, cfg.TEXT_COLOR, cfg.THICKNESS)
        cv2.putText(frame, f"Gesture: {inference_res['gesture']}", (25, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, cfg.FONT_SCALE, cfg.HIGHLIGHT_COLOR, cfg.THICKNESS)
        cv2.putText(frame, f"Robot Action: {inference_res['action']}", (25, 185), 
                    cv2.FONT_HERSHEY_SIMPLEX, cfg.FONT_SCALE, (0, 255, 255), cfg.THICKNESS)

        # Show Window
        cv2.imshow("Robot Vision Control Center", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
