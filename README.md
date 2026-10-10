# 🚗 Virtual Steering Wheel

Control any car game with your hands and webcam — no keyboard, no controller needed.

Uses **MediaPipe** hand tracking to detect steering angle and hand gestures for acceleration and braking.

---

## 🎮 Controls

| Action | How |
|---|---|
| Steer | Tilt both hands left / right like a wheel |
| Accelerate | Make a **fist** (or hold neutral) |
| Brake | Open your hand flat |

Outputs to both **Xbox virtual controller** (analog stick + triggers) and **W / S keys**.

---

## ⚙️ Requirements

- Windows 10/11
- Python 3.11+
- Webcam
- A racing game that supports keyboard or Xbox controller

---

## 🚀 Setup

```bash
git clone https://github.com/samyakjain8431-cyber/Virtual-steering-wheel.git
cd Virtual-steering-wheel

python -m venv venv
venv\Scripts\Activate.ps1

pip install opencv-python mediapipe vgamepad pynput
```

> Make sure `hand_landmarker.task` is in the project folder.

---

## ▶️ Run

```bash
python main.py
```

Press **`Q`** on the webcam window to quit.

---

## 📁 Project Structure

```
├── main.py          # Main loop
├── config.py        # Constants & settings
├── steering.py      # Angle → analog stick mapping
├── gesture.py       # Hand gesture detection
├── drawing.py       # Webcam overlay & HUD
├── controller.py    # Gamepad & keyboard output
├── tracker.py       # MediaPipe setup
└── hand_landmarker.task
```

---

## 👨‍💻 Author

**Samyak Jain** — built as a computer-vision experiment with hand tracking and game control.
