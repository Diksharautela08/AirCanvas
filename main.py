import cv2
import mediapipe as mp
import numpy as np
import time
from pathlib import Path
from datetime import datetime

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# AIRCanvas PRO
# ============================================================

WINDOW_NAME = "AirCanvas Pro"

# ------------------------------------------------------------
# PROJECT PATHS
# ------------------------------------------------------------

PROJECT_FOLDER = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_FOLDER / "models" / "hand_landmarker.task"
SAVE_FOLDER = PROJECT_FOLDER / "saved_drawings"


# ============================================================
# SETTINGS
# ============================================================

SMOOTHING = 0.28

BRUSH_SIZE = 8
ERASER_SIZE = 35

GESTURE_CONFIRM_FRAMES = 4
MAX_HISTORY = 20

SELECTION_HOLD_SECONDS = 1.0


# ============================================================
# GESTURE STABILIZATION
# ============================================================

stable_gesture = "NO HAND"

candidate_gesture = None
candidate_frames = 0


# ============================================================
# UNDO / REDO
# ============================================================

undo_stack = []
redo_stack = []


# ============================================================
# BRUSH / ERASER OPTIONS
# ============================================================

BRUSH_SIZE_OPTIONS = [
    (4, "Small"),
    (8, "Medium"),
    (16, "Large"),
]

ERASER_SIZE_OPTIONS = [
    (20, "Small"),
    (35, "Medium"),
    (55, "Large"),
]


# ============================================================
# COLORS
# OpenCV uses BGR format
# ============================================================

COLORS = {
    "Purple": (255, 0, 255),
    "Red": (0, 0, 255),
    "Green": (0, 255, 0),
    "Blue": (255, 0, 0),
    "Yellow": (0, 255, 255),
}

selected_color_name = "Purple"
BRUSH_COLOR = COLORS[selected_color_name]


# ============================================================
# SETTINGS PANEL
# ============================================================

settings_open = False

SETTINGS_X = 15
SETTINGS_Y = 70
SETTINGS_WIDTH = 220
SETTINGS_HEIGHT = 300

TOGGLE_X1 = 15
TOGGLE_Y1 = 70
TOGGLE_X2 = 145
TOGGLE_Y2 = 110

COLOR_Y = 130
BRUSH_Y = 205
ERASER_Y = 260
HISTORY_Y = 315

BUTTON_WIDTH = 60
BUTTON_HEIGHT = 35
BUTTON_GAP = 8


# ============================================================
# COLOR SELECTION STATE
# ============================================================

selection_start_time = None
hovered_color_name = None
selection_locked = False

color_message = ""
color_message_until = 0


# ============================================================
# SAVE MESSAGE
# ============================================================

save_message = ""
save_message_until = 0


# ============================================================
# DRAWING STATE
# ============================================================

gesture = "NO HAND"

current_history_action = None

canvas = None


# ============================================================
# UI BUTTON STORAGE
# ============================================================

color_buttons = []
brush_buttons = []
eraser_buttons = []


# ============================================================
# GESTURE STABILIZATION
# ============================================================

def stabilize_gesture(raw_gesture):
    """
    Prevents accidental gesture switching by requiring
    a gesture to remain stable for several frames.
    """

    global stable_gesture
    global candidate_gesture
    global candidate_frames

    # No hand should immediately reset the state
    if raw_gesture == "NO HAND":

        stable_gesture = "NO HAND"
        candidate_gesture = None
        candidate_frames = 0

        return "NO HAND"

    # Same as current stable gesture
    if raw_gesture == stable_gesture:

        candidate_gesture = None
        candidate_frames = 0

        return stable_gesture

    # New candidate gesture
    if raw_gesture != candidate_gesture:

        candidate_gesture = raw_gesture
        candidate_frames = 1

        return "TRANSITION"

    # Candidate continues
    candidate_frames += 1

    # Confirm gesture
    if candidate_frames >= GESTURE_CONFIRM_FRAMES:

        stable_gesture = candidate_gesture

        candidate_gesture = None
        candidate_frames = 0

        return stable_gesture

    return "TRANSITION"


# ============================================================
# UNDO / REDO
# ============================================================

def save_undo_state():
    """
    Saves the current canvas state before a new action.
    """

    global undo_stack
    global redo_stack

    if canvas is None:
        return

    undo_stack.append(canvas.copy())

    if len(undo_stack) > MAX_HISTORY:
        undo_stack.pop(0)

    redo_stack.clear()


def undo_action():
    """
    Undo the most recent drawing action.
    """

    global canvas
    global undo_stack
    global redo_stack

    if canvas is None:
        return

    if len(undo_stack) == 0:
        return

    redo_stack.append(canvas.copy())

    canvas = undo_stack.pop()


def redo_action():
    """
    Redo the most recently undone action.
    """

    global canvas
    global undo_stack
    global redo_stack

    if canvas is None:
        return

    if len(redo_stack) == 0:
        return

    undo_stack.append(canvas.copy())

    canvas = redo_stack.pop()


# ============================================================
# SAVE ARTWORK
# ============================================================

def save_artwork(canvas_image):
    """
    Saves only the drawing canvas as a PNG.
    Webcam feed and UI are not saved.
    """

    global save_message
    global save_message_until

    if canvas_image is None:
        return

    SAVE_FOLDER.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    filename = f"aircanvas_{timestamp}.png"

    save_path = SAVE_FOLDER / filename

    success = cv2.imwrite(
        str(save_path),
        canvas_image
    )

    if success:

        save_message = f"Saved: {filename}"

        save_message_until = time.time() + 3

        print(f"Artwork saved to: {save_path}")

    else:

        save_message = "Save failed"

        save_message_until = time.time() + 3


# ============================================================
# GESTURE DETECTION
# ============================================================

def detect_gesture(hand_landmarks):
    """
    Detects basic drawing gestures.

    INDEX ONLY
        -> DRAW

    INDEX + MIDDLE
        -> SELECT

    FOUR FINGERS
        -> ERASER

    OTHERWISE
        -> NONE
    """

    # MediaPipe landmark indices
    # 8  = index tip
    # 12 = middle tip
    # 16 = ring tip
    # 20 = pinky tip

    index_up = (
        hand_landmarks[8].y
        < hand_landmarks[6].y
    )

    middle_up = (
        hand_landmarks[12].y
        < hand_landmarks[10].y
    )

    ring_up = (
        hand_landmarks[16].y
        < hand_landmarks[14].y
    )

    pinky_up = (
        hand_landmarks[20].y
        < hand_landmarks[18].y
    )

    # Four fingers
    if (
        index_up
        and middle_up
        and ring_up
        and pinky_up
    ):
        return "ERASER"

    # Index + middle
    if index_up and middle_up:
        return "SELECT"

    # Index only
    if index_up and not middle_up:
        return "DRAW"

    return "NONE"


# ============================================================
# SETTINGS PANEL
# ============================================================

def draw_settings_panel(display):

    global color_buttons
    global brush_buttons
    global eraser_buttons

    # Reset button lists every frame
    color_buttons = []
    brush_buttons = []
    eraser_buttons = []

    # --------------------------------------------------------
    # SETTINGS TOGGLE BUTTON
    # --------------------------------------------------------

    if not settings_open:

        cv2.rectangle(
            display,
            (TOGGLE_X1, TOGGLE_Y1),
            (TOGGLE_X2, TOGGLE_Y2),
            (40, 40, 40),
            -1
        )

        cv2.rectangle(
            display,
            (TOGGLE_X1, TOGGLE_Y1),
            (TOGGLE_X2, TOGGLE_Y2),
            (255, 255, 255),
            1
        )

        cv2.putText(
            display,
            "SETTINGS",
            (TOGGLE_X1 + 15, TOGGLE_Y1 + 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        return

    # --------------------------------------------------------
    # PANEL
    # --------------------------------------------------------

    panel_x1 = SETTINGS_X
    panel_y1 = SETTINGS_Y

    panel_x2 = SETTINGS_X + SETTINGS_WIDTH
    panel_y2 = SETTINGS_Y + SETTINGS_HEIGHT

    # Slight transparent-looking dark panel
    overlay = display.copy()

    cv2.rectangle(
        overlay,
        (panel_x1, panel_y1),
        (panel_x2, panel_y2),
        (25, 25, 25),
        -1
    )

    cv2.addWeighted(
        overlay,
        0.90,
        display,
        0.10,
        0,
        display
    )

    cv2.rectangle(
        display,
        (panel_x1, panel_y1),
        (panel_x2, panel_y2),
        (255, 255, 255),
        1
    )

    # --------------------------------------------------------
    # PANEL TITLE
    # --------------------------------------------------------

    cv2.putText(
        display,
        "SETTINGS",
        (panel_x1 + 12, panel_y1 + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    # Close button
    close_x1 = panel_x2 - 38
    close_y1 = panel_y1 + 8
    close_x2 = panel_x2 - 8
    close_y2 = panel_y1 + 38

    cv2.rectangle(
        display,
        (close_x1, close_y1),
        (close_x2, close_y2),
        (70, 70, 70),
        -1
    )

    cv2.putText(
        display,
        "X",
        (close_x1 + 8, close_y1 + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    # --------------------------------------------------------
    # COLORS
    # --------------------------------------------------------

    cv2.putText(
        display,
        "COLORS",
        (panel_x1 + 12, COLOR_Y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 200),
        1,
        cv2.LINE_AA
    )

    color_names = list(COLORS.keys())

    color_size = 30
    color_gap = 8

    start_x = panel_x1 + 12

    for i, name in enumerate(color_names):

        x1 = start_x + i * (color_size + color_gap)
        y1 = COLOR_Y
        x2 = x1 + color_size
        y2 = y1 + color_size

        # Store clickable region
        color_buttons.append(
            (name, x1, y1, x2, y2)
        )

        # Color square
        cv2.rectangle(
            display,
            (x1, y1),
            (x2, y2),
            COLORS[name],
            -1
        )

        # Selected color border
        if name == selected_color_name:

            cv2.rectangle(
                display,
                (x1 - 3, y1 - 3),
                (x2 + 3, y2 + 3),
                (255, 255, 255),
                2
            )

    # Instruction
    cv2.putText(
        display,
        "Click color / use SELECT",
        (panel_x1 + 12, COLOR_Y + 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (180, 180, 180),
        1,
        cv2.LINE_AA
    )

    # --------------------------------------------------------
    # BRUSH SIZE
    # --------------------------------------------------------

    cv2.putText(
        display,
        "BRUSH SIZE",
        (panel_x1 + 12, BRUSH_Y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 200),
        1,
        cv2.LINE_AA
    )

    brush_start_x = panel_x1 + 12

    for i, (size, label) in enumerate(BRUSH_SIZE_OPTIONS):

        x1 = brush_start_x + i * (
            BUTTON_WIDTH + BUTTON_GAP
        )

        y1 = BRUSH_Y
        x2 = x1 + BUTTON_WIDTH
        y2 = y1 + BUTTON_HEIGHT

        brush_buttons.append(
            (size, x1, y1, x2, y2)
        )

        selected = (
            size == BRUSH_SIZE
        )

        cv2.rectangle(
            display,
            (x1, y1),
            (x2, y2),
            (70, 70, 70),
            -1
        )

        cv2.rectangle(
            display,
            (x1, y1),
            (x2, y2),
            (255, 255, 255) if selected
            else (100, 100, 100),
            2 if selected else 1
        )

        cv2.putText(
            display,
            label,
            (x1 + 7, y1 + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    # --------------------------------------------------------
    # ERASER SIZE
    # --------------------------------------------------------

    cv2.putText(
        display,
        "ERASER SIZE",
        (panel_x1 + 12, ERASER_Y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 200),
        1,
        cv2.LINE_AA
    )

    eraser_start_x = panel_x1 + 12

    for i, (size, label) in enumerate(
        ERASER_SIZE_OPTIONS
    ):

        x1 = eraser_start_x + i * (
            BUTTON_WIDTH + BUTTON_GAP
        )

        y1 = ERASER_Y
        x2 = x1 + BUTTON_WIDTH
        y2 = y1 + BUTTON_HEIGHT

        eraser_buttons.append(
            (size, x1, y1, x2, y2)
        )

        selected = (
            size == ERASER_SIZE
        )

        cv2.rectangle(
            display,
            (x1, y1),
            (x2, y2),
            (70, 70, 70),
            -1
        )

        cv2.rectangle(
            display,
            (x1, y1),
            (x2, y2),
            (255, 255, 255) if selected
            else (100, 100, 100),
            2 if selected else 1
        )

        cv2.putText(
            display,
            label,
            (x1 + 7, y1 + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    # --------------------------------------------------------
    # HISTORY
    # --------------------------------------------------------

    cv2.putText(
        display,
        "HISTORY",
        (panel_x1 + 12, HISTORY_Y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 200),
        1,
        cv2.LINE_AA
    )

    undo_x1 = panel_x1 + 12
    undo_y1 = HISTORY_Y
    undo_x2 = undo_x1 + 88
    undo_y2 = undo_y1 + 32

    redo_x1 = undo_x2 + 8
    redo_y1 = HISTORY_Y
    redo_x2 = redo_x1 + 88
    redo_y2 = redo_y1 + 32

    cv2.rectangle(
        display,
        (undo_x1, undo_y1),
        (undo_x2, undo_y2),
        (70, 70, 70),
        -1
    )

    cv2.rectangle(
        display,
        (redo_x1, redo_y1),
        (redo_x2, redo_y2),
        (70, 70, 70),
        -1
    )

    cv2.putText(
        display,
        "UNDO",
        (undo_x1 + 20, undo_y1 + 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    cv2.putText(
        display,
        "REDO",
        (redo_x1 + 20, redo_y1 + 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )


# ============================================================
# MOUSE CLICK HANDLER
# ============================================================

def handle_mouse_click(event, x, y, flags, param):

    global BRUSH_SIZE
    global ERASER_SIZE
    global settings_open

    global selected_color_name
    global BRUSH_COLOR

    global color_message
    global color_message_until

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    # --------------------------------------------------------
    # SETTINGS TOGGLE
    # --------------------------------------------------------

    if (
        TOGGLE_X1 <= x <= TOGGLE_X2
        and TOGGLE_Y1 <= y <= TOGGLE_Y2
    ):

        settings_open = not settings_open

        return

    # If settings are closed, nothing else is clickable
    if not settings_open:
        return

    # --------------------------------------------------------
    # CLOSE BUTTON
    # --------------------------------------------------------

    panel_x2 = SETTINGS_X + SETTINGS_WIDTH

    close_x1 = panel_x2 - 38
    close_y1 = SETTINGS_Y + 8
    close_x2 = panel_x2 - 8
    close_y2 = SETTINGS_Y + 38

    if (
        close_x1 <= x <= close_x2
        and close_y1 <= y <= close_y2
    ):

        settings_open = False

        return

    # --------------------------------------------------------
    # COLOR SELECTION
    # --------------------------------------------------------

    for name, x1, y1, x2, y2 in color_buttons:

        if (
            x1 <= x <= x2
            and y1 <= y <= y2
        ):

            selected_color_name = name
            BRUSH_COLOR = COLORS[name]

            color_message = f"{name} selected"
            color_message_until = time.time() + 2

            print(f"Color selected: {name}")

            return

    # --------------------------------------------------------
    # BRUSH SIZE
    # --------------------------------------------------------

    for size, x1, y1, x2, y2 in brush_buttons:

        if (
            x1 <= x <= x2
            and y1 <= y <= y2
        ):

            BRUSH_SIZE = size

            color_message = (
                f"Brush size: {size}"
            )

            color_message_until = time.time() + 2

            print(f"Brush size: {size}")

            return

    # --------------------------------------------------------
    # ERASER SIZE
    # --------------------------------------------------------

    for size, x1, y1, x2, y2 in eraser_buttons:

        if (
            x1 <= x <= x2
            and y1 <= y <= y2
        ):

            ERASER_SIZE = size

            color_message = (
                f"Eraser size: {size}"
            )

            color_message_until = time.time() + 2

            print(f"Eraser size: {size}")

            return

    # --------------------------------------------------------
    # UNDO
    # --------------------------------------------------------

    undo_x1 = SETTINGS_X + 12
    undo_y1 = HISTORY_Y
    undo_x2 = undo_x1 + 88
    undo_y2 = undo_y1 + 32

    if (
        undo_x1 <= x <= undo_x2
        and undo_y1 <= y <= undo_y2
    ):

        undo_action()

        return

    # --------------------------------------------------------
    # REDO
    # --------------------------------------------------------

    redo_x1 = undo_x2 + 8
    redo_y1 = HISTORY_Y
    redo_x2 = redo_x1 + 88
    redo_y2 = redo_y1 + 32

    if (
        redo_x1 <= x <= redo_x2
        and redo_y1 <= y <= redo_y2
    ):

        redo_action()

        return


# ============================================================
# CHECK MODEL
# ============================================================

if not MODEL_PATH.exists():

    print()
    print("ERROR: MediaPipe hand landmark model not found.")
    print()
    print("Expected location:")
    print(MODEL_PATH)
    print()

    raise SystemExit


# ============================================================
# MEDIAPIPE HAND LANDMARKER
# ============================================================

base_options = python.BaseOptions(
    model_asset_path=str(MODEL_PATH)
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7
)

detector = vision.HandLandmarker.create_from_options(
    options
)


# ============================================================
# WEBCAM
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("Could not access webcam.")

    detector.close()

    raise SystemExit


# ============================================================
# INITIALIZE
# ============================================================

canvas = None

previous_point = None
smoothed_point = None

timestamp_ms = 0

cv2.namedWindow(WINDOW_NAME)

cv2.setMouseCallback(
    WINDOW_NAME,
    handle_mouse_click
)


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    success, frame = cap.read()

    if not success:
        print("Could not read webcam frame.")
        break

    # --------------------------------------------------------
    # MIRROR CAMERA
    # --------------------------------------------------------

    frame = cv2.flip(frame, 1)

    height, width = frame.shape[:2]

    # --------------------------------------------------------
    # CREATE CANVAS
    # --------------------------------------------------------

    if canvas is None:

        canvas = np.zeros_like(frame)

    # --------------------------------------------------------
    # MEDIAPIPE INPUT
    # --------------------------------------------------------

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    timestamp_ms += 33

    result = detector.detect_for_video(
        mp_image,
        timestamp_ms
    )

    # ========================================================
    # HAND DETECTED
    # ========================================================

    if result.hand_landmarks:

        hand_landmarks = result.hand_landmarks[0]

        # ----------------------------------------------------
        # RAW GESTURE
        # ----------------------------------------------------

        raw_gesture = detect_gesture(
            hand_landmarks
        )

        # ----------------------------------------------------
        # STABILIZED GESTURE
        # ----------------------------------------------------

        gesture = stabilize_gesture(
            raw_gesture
        )

        # ----------------------------------------------------
        # INDEX FINGER TIP
        # ----------------------------------------------------

        index_tip = hand_landmarks[8]

        tip_x = int(
            index_tip.x * width
        )

        tip_y = int(
            index_tip.y * height
        )

        # Keep coordinates inside frame
        tip_x = max(
            0,
            min(width - 1, tip_x)
        )

        tip_y = max(
            0,
            min(height - 1, tip_y)
        )

        # ====================================================
        # TRANSITION
        # ====================================================

        if gesture == "TRANSITION":

            previous_point = None
            smoothed_point = None

            selection_start_time = None
            hovered_color_name = None

            selection_locked = False

        # ====================================================
        # SELECT GESTURE
        # ====================================================

        elif gesture == "SELECT":

            # Drawing must stop
            previous_point = None
            smoothed_point = None

            # ------------------------------------------------
            # COLOR SELECTION ONLY WHEN SETTINGS ARE OPEN
            # ------------------------------------------------

            if settings_open:

                current_hover = None

                for (
                    name,
                    x1,
                    y1,
                    x2,
                    y2
                ) in color_buttons:

                    if (
                        x1 <= tip_x <= x2
                        and y1 <= tip_y <= y2
                    ):

                        current_hover = name
                        break

                # --------------------------------------------
                # Finger moved to another color
                # --------------------------------------------

                if current_hover != hovered_color_name:

                    hovered_color_name = current_hover
                    selection_start_time = time.time()
                    selection_locked = False

                # --------------------------------------------
                # Holding over a color
                # --------------------------------------------

                if (
                    current_hover is not None
                    and not selection_locked
                    and selection_start_time is not None
                ):

                    elapsed = (
                        time.time()
                        - selection_start_time
                    )

                    progress = min(
                        elapsed / SELECTION_HOLD_SECONDS,
                        1.0
                    )

                    # Progress bar
                    bar_x1 = 15
                    bar_y1 = 385
                    bar_x2 = 215
                    bar_y2 = 400

                    cv2.rectangle(
                        frame,
                        (bar_x1, bar_y1),
                        (bar_x2, bar_y2),
                        (60, 60, 60),
                        -1
                    )

                    cv2.rectangle(
                        frame,
                        (bar_x1, bar_y1),
                        (
                            bar_x1
                            + int(
                                (bar_x2 - bar_x1)
                                * progress
                            ),
                            bar_y2
                        ),
                        COLORS[current_hover],
                        -1
                    )

                    cv2.putText(
                        frame,
                        f"Selecting {current_hover}...",
                        (15, 425),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA
                    )

                    # ----------------------------------------
                    # Selection complete
                    # ----------------------------------------

                    if progress >= 1.0:

                        selected_color_name = current_hover

                        BRUSH_COLOR = COLORS[
                            current_hover
                        ]

                        color_message = (
                            f"{current_hover} selected"
                        )

                        color_message_until = (
                            time.time() + 2
                        )

                        selection_locked = True

                        print(
                            f"Gesture color selected: "
                            f"{current_hover}"
                        )

                # Cursor
                cv2.circle(
                    frame,
                    (tip_x, tip_y),
                    10,
                    (255, 255, 255),
                    2
                )

            else:

                cv2.putText(
                    frame,
                    "Open SETTINGS to select color",
                    (20, height - 85),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

        # ====================================================
        # DRAW
        # ====================================================

        elif gesture == "DRAW":

            # ------------------------------------------------
            # Save history at beginning of new draw action
            # ------------------------------------------------

            if current_history_action != "DRAW":

                save_undo_state()

                current_history_action = "DRAW"

            # ------------------------------------------------
            # SMOOTH POSITION
            # ------------------------------------------------

            if smoothed_point is None:

                smoothed_point = (
                    tip_x,
                    tip_y
                )

            else:

                smooth_x = (
                    SMOOTHING * tip_x
                    + (1 - SMOOTHING)
                    * smoothed_point[0]
                )

                smooth_y = (
                    SMOOTHING * tip_y
                    + (1 - SMOOTHING)
                    * smoothed_point[1]
                )

                smoothed_point = (
                    int(smooth_x),
                    int(smooth_y)
                )

            current_point = smoothed_point

            # ------------------------------------------------
            # DRAW LINE
            # ------------------------------------------------

            if previous_point is not None:

                cv2.line(
                    canvas,
                    previous_point,
                    current_point,
                    BRUSH_COLOR,
                    BRUSH_SIZE,
                    cv2.LINE_AA
                )

            previous_point = current_point

            # Drawing cursor
            cv2.circle(
                frame,
                current_point,
                max(3, BRUSH_SIZE // 2),
                BRUSH_COLOR,
                2
            )

        # ====================================================
        # ERASER
        # ====================================================

        elif gesture == "ERASER":

            # ------------------------------------------------
            # Save history at beginning of new eraser action
            # ------------------------------------------------

            if current_history_action != "ERASER":

                save_undo_state()

                current_history_action = "ERASER"

            # ------------------------------------------------
            # Smooth position
            # ------------------------------------------------

            if smoothed_point is None:

                smoothed_point = (
                    tip_x,
                    tip_y
                )

            else:

                smooth_x = (
                    SMOOTHING * tip_x
                    + (1 - SMOOTHING)
                    * smoothed_point[0]
                )

                smooth_y = (
                    SMOOTHING * tip_y
                    + (1 - SMOOTHING)
                    * smoothed_point[1]
                )

                smoothed_point = (
                    int(smooth_x),
                    int(smooth_y)
                )

            current_point = smoothed_point

            # ------------------------------------------------
            # Erase
            # ------------------------------------------------

            cv2.circle(
                canvas,
                current_point,
                ERASER_SIZE,
                (0, 0, 0),
                -1
            )

            previous_point = current_point

            # Eraser cursor
            cv2.circle(
                frame,
                current_point,
                ERASER_SIZE,
                (255, 255, 255),
                2
            )

        # ====================================================
        # NONE
        # ====================================================

        elif gesture == "NONE":

            previous_point = None
            smoothed_point = None

            selection_start_time = None
            hovered_color_name = None
            selection_locked = False

            current_history_action = None

    # ========================================================
    # NO HAND
    # ========================================================

    else:

        gesture = stabilize_gesture(
            "NO HAND"
        )

        previous_point = None
        smoothed_point = None

        selection_start_time = None
        hovered_color_name = None
        selection_locked = False

        current_history_action = None

    # ========================================================
    # COMBINE CAMERA + CANVAS
    # ========================================================

    display = cv2.add(
        frame,
        canvas
    )

    # ========================================================
    # TOP HEADER
    # ========================================================

    cv2.rectangle(
        display,
        (0, 0),
        (width, 55),
        (25, 25, 25),
        -1
    )

    cv2.putText(
        display,
        "AIRCanvas PRO",
        (15, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    # --------------------------------------------------------
    # GESTURE STATUS
    # --------------------------------------------------------

    status_text = f"Gesture: {gesture}"

    cv2.putText(
        display,
        status_text,
        (width - 240, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    # ========================================================
    # SETTINGS PANEL
    # ========================================================

    draw_settings_panel(display)

    # ========================================================
    # CURRENT COLOR INDICATOR
    # ========================================================

    indicator_x = width - 60
    indicator_y = 70

    cv2.circle(
        display,
        (indicator_x, indicator_y),
        18,
        BRUSH_COLOR,
        -1
    )

    cv2.circle(
        display,
        (indicator_x, indicator_y),
        20,
        (255, 255, 255),
        2
    )

    # ========================================================
    # TRANSITION MESSAGE
    # ========================================================

    if gesture == "TRANSITION":

        cv2.putText(
            display,
            "Hold gesture steady...",
            (width // 2 - 120, height - 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    # ========================================================
    # COLOR / SIZE MESSAGE
    # ========================================================

    if (
        color_message
        and time.time() < color_message_until
    ):

        cv2.putText(
            display,
            color_message,
            (width // 2 - 90, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    # ========================================================
    # SAVE MESSAGE
    # ========================================================

    if (
        save_message
        and time.time() < save_message_until
    ):

        cv2.putText(
            display,
            save_message,
            (20, height - 125),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    # ========================================================
    # BOTTOM CONTROLS
    # ========================================================

    cv2.putText(
        display,
        "DRAW: Index | SELECT: Index+Middle | "
        "ERASE: 4 Fingers",
        (20, height - 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (220, 220, 220),
        1,
        cv2.LINE_AA
    )

    cv2.putText(
        display,
        "S: Save   C: Clear   U: Undo   R: Redo   Q: Quit",
        (20, height - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (220, 220, 220),
        1,
        cv2.LINE_AA
    )

    # ========================================================
    # SHOW
    # ========================================================

    cv2.imshow(
        WINDOW_NAME,
        display
    )

    # ========================================================
    # KEYBOARD CONTROLS
    # ========================================================

    key = cv2.waitKey(1) & 0xFF

    # --------------------------------------------------------
    # QUIT
    # --------------------------------------------------------

    if key == ord("q"):

        break

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    elif key == ord("s"):

        save_artwork(canvas)

    # --------------------------------------------------------
    # CLEAR
    # --------------------------------------------------------

    elif key == ord("c"):

        save_undo_state()

        canvas[:] = 0

        previous_point = None
        smoothed_point = None

        current_history_action = None

    # --------------------------------------------------------
    # UNDO
    # --------------------------------------------------------

    elif key == ord("u") or key == 26:

        undo_action()

        previous_point = None
        smoothed_point = None

        current_history_action = None

    # --------------------------------------------------------
    # REDO
    # --------------------------------------------------------

    elif key == ord("r") or key == 25:

        redo_action()

        previous_point = None
        smoothed_point = None

        current_history_action = None


# ============================================================
# CLEANUP
# ============================================================

cap.release()

detector.close()

cv2.destroyAllWindows()