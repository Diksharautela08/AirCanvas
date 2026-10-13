# AirCanvas Pro — Touchless Virtual Drawing Using Computer Vision

## 1. Project Overview

AirCanvas Pro is a computer vision mini-project that allows users to draw on a virtual canvas using hand gestures in front of a webcam. It uses real-time hand tracking to recognize finger positions and translate them into drawing actions.

The application provides a touchless drawing experience without requiring a physical mouse, touchscreen, or drawing tablet for drawing. Mouse clicks are used for settings and color selection.

## 2. Problem Statement

Traditional digital drawing applications generally require a mouse, stylus, or touchscreen. This project explores an alternative approach in which hand gestures captured through a webcam are used to control drawing operations.

## 3. Objectives

- Implement real-time hand tracking using a webcam.
- Recognize hand gestures for drawing, selection, and erasing.
- Create a virtual drawing canvas using OpenCV.
- Provide multiple colors and adjustable brush and eraser sizes.
- Implement undo, redo, clear, and save functionality.
- Improve drawing stability using gesture confirmation and position smoothing.

## 4. Features

- **Real-time hand tracking:** Tracks hand landmarks using MediaPipe.
- **Gesture-based drawing:** Uses the index finger to draw.
- **Gesture-based selection:** Uses the index and middle fingers together.
- **Gesture-based erasing:** Uses four raised fingers.
- **Multiple colors:** Purple, red, green, blue, and yellow.
- **Adjustable brush size:** Small, medium, and large.
- **Adjustable eraser size:** Small, medium, and large.
- **Collapsible settings panel:** Keeps the drawing area clear when settings are not needed.
- **Undo and redo:** Allows previous drawing actions to be reversed or restored.
- **Clear canvas:** Removes the current drawing.
- **Save artwork:** Exports the drawing as a PNG image.
- **Gesture stabilization:** Requires a gesture to remain consistent across multiple frames.
- **Position smoothing:** Reduces sudden movement in the drawing cursor.

## 5. Technologies Used

- **Python:** Main programming language.
- **OpenCV:** Webcam capture, image processing, drawing operations, and user interface.
- **MediaPipe Tasks:** Real-time hand landmark detection.
- **NumPy:** Canvas and image-array operations.

## 6. How It Works

1. The webcam captures live video.
2. OpenCV reads and mirrors the video frames.
3. MediaPipe detects the hand and identifies its landmarks.
4. The application checks the positions of selected finger landmarks to recognize gestures.
5. A gesture stabilization mechanism reduces accidental switching between gestures.
6. The index fingertip position is smoothed to make drawing more stable.
7. OpenCV draws colored lines on a separate virtual canvas.
8. The virtual canvas is combined with the webcam feed for display.
9. Users can manage settings, undo or redo actions, clear the canvas, and save their artwork.

## 7. Gesture Controls

| Hand gesture | Action |
|---|---|
| Index finger raised, middle finger lowered | Draw |
| Index and middle fingers raised | Select a color when settings are open |
| Four fingers raised | Erase |
| Other recognized hand poses | No drawing action |

**Note:** Gesture recognition depends on visible hand landmarks, hand orientation, lighting, and camera positioning.

## 8. Keyboard Controls

| Key | Action |
|---|---|
| S | Save artwork |
| C | Clear canvas |
| U | Undo |
| R | Redo |
| Q | Quit application |

## 9. Project Structure

```text
AirCanvas-Pro/
│
├── main.py
├── requirements.txt
├── README.md
│
├── models/
│   └── hand_landmarker.task
│
├── saved_drawings/
   └── Generated PNG artwork.

## 10. Installation and Setup

### Prerequisites

- Python installed on your computer.
- A working webcam.
- The MediaPipe Hand Landmarker model file.

### Step 1: Clone the repository

```bash
git clone https://github.com/Diksharautela08/AirCanvas
cd AirCanvas-Pro
```

### Step 2: Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Step 3: Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### Step 4: Download the hand landmark model

Place the official MediaPipe Hand Landmarker model file at:

```text
models/hand_landmarker.task
```

The application expects this exact path relative to `main.py`.

### Step 5: Run the application

```powershell
python main.py
```

Allow webcam access if requested.

## 11. Expected Results

When the application runs successfully:

- The webcam feed appears in a window.
- A detected hand can control drawing through gestures.
- The selected color and brush size determine the appearance of strokes.
- The eraser removes portions of the virtual drawing.
- Undo and redo restore previous canvas states.
- Saved artwork is exported as a PNG image in the `saved_drawings/` directory.

Actual performance depends on lighting, camera quality, and how clearly the hand is visible.

## 12. Limitations

- Requires a functioning webcam.
- Hand tracking may become less reliable in poor lighting or when fingers are occluded.
- The implemented gestures are limited to a predefined set.
- Gesture-based color selection requires the settings panel to be open.
- The current application is a local desktop prototype rather than a deployed web application.

## 13. Future Scope

Possible future improvements include multi-hand support, additional gestures, customizable shortcuts, configurable canvas backgrounds, and a more advanced drawing interface.

## 14. Conclusion

AirCanvas Pro demonstrates how computer vision can be used to create a touchless drawing interface. By combining hand landmark detection, gesture recognition, image processing, and a virtual canvas, the project provides an interactive application controlled primarily through hand movements.

## 15. Acknowledgements

This project uses Python, OpenCV, and MediaPipe. Refer to their official documentation for technical details and licensing information.
