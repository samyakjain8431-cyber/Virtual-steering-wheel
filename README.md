# Virtual Steering Wheel
# 🚗 Virtual Steering Wheel

A computer-vision-based **virtual steering wheel** that lets you control a car game using your **hands and webcam**.

Instead of using a physical keyboard or steering wheel, the project uses **MediaPipe hand tracking** to detect your hands and determine the steering direction. Keyboard inputs are then sent to the game using `pydirectinput`.

## ✨ Features

* 🎥 Uses your webcam for real-time hand tracking
* ✋ Detects both hands using MediaPipe
* 🔄 Calculates the relative position/angle of your hands
* ⬅️ Detects **Left**, **Right**, and **Straight**
* 🎮 Sends keyboard controls to the game automatically
* 🐍 Built completely with Python

---

# 🛠️ Requirements

Before starting, make sure you have:

* Windows 10/11
* Python 3.11 recommended
* A working webcam
* A car/racing game that accepts keyboard controls
* Git installed

---

# 📥 Installation

## 1. Clone the repository

Open **Command Prompt** or **PowerShell** and run:

```bash
git clone https://github.com/samyakjain8431-cyber/Virtual-steering-wheel.git
```

Move into the project folder:

```bash
cd Virtual-steering-wheel
```

---

## 2. Create a virtual environment

It is recommended to use a virtual environment so the project's packages don't interfere with other Python projects.

```bash
python -m venv .venv
```

Activate it:

### Windows PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

### Windows Command Prompt

```cmd
.venv\Scripts\activate
```

After activation, you should see something like:

```text
(.venv)
```

at the beginning of your terminal line.

---

# 📦 Install Required Packages

Install the required Python packages:

```bash
pip install opencv-python mediapipe pydirectinput
```

You can verify the installation with:

```bash
pip list
```

You should see packages such as:

```text
opencv-python
mediapipe
pydirectinput
```

---

# 🤖 MediaPipe Model

This project requires the MediaPipe hand-landmark model:

```text
hand_landmarker.task
```

The model file should be located inside the project folder:

```text
Virtual-steering-wheel/
│
├── main.py
├── hand_landmarker.task
├── README.md
└── ...
```

If the file is already included in the repository, you don't need to download it separately.

---

# ▶️ Run the Project

Make sure your virtual environment is activated.

Then run:

```bash
python main.py
```

A webcam window should open.

Position both hands in front of the camera as if you are holding a steering wheel.

The program will track your hands and determine whether you are turning:

```text
LEFT
STRAIGHT
RIGHT
```

---

# 🎮 Using It With a Game

1. Start the project:

```bash
python main.py
```

2. Allow the webcam to detect your hands.

3. Position your hands like you are holding a steering wheel.

4. Start your car game.

5. Move your hands left or right to control the vehicle.

The program sends keyboard inputs automatically.

### Current Controls

| Hand Movement       | Game Input      |
| ------------------- | --------------- |
| Turn left           | `A`             |
| Keep hands straight | No steering key |
| Turn right          | `D`             |

> The exact behavior depends on how the game maps its keyboard controls.

---

# 🖐️ Recommended Hand Position

For better detection:

* Keep both hands visible to the webcam.
* Keep your hands reasonably separated.
* Use good lighting.
* Avoid covering your hands.
* Keep the camera stable.
* Sit at a suitable distance from the webcam.

Think of your hands as the two sides of an invisible steering wheel.

---

# ⚠️ Troubleshooting

## Camera doesn't open

Check whether another application is already using your webcam.

Close applications such as:

* Camera
* Microsoft Teams
* Discord
* Zoom
* Other Python programs using the webcam

Then run:

```bash
python main.py
```

---

## `ModuleNotFoundError`

For example:

```text
ModuleNotFoundError: No module named 'mediapipe'
```

Make sure your virtual environment is activated and reinstall the packages:

```bash
pip install opencv-python mediapipe pydirectinput
```

---

## `git` is not recognized

If you see:

```text
'git' is not recognized as an internal or external command
```

Git is either not installed or isn't available in your PATH.

Install Git and restart your terminal.

---

## Hands are not detected properly

Try:

* Increasing the lighting
* Moving closer/farther from the camera
* Keeping both hands completely inside the camera frame
* Removing objects that block your hands
* Keeping the background reasonably clear

---

## Program does not quit with `q`

Make sure the webcam window is selected/focused and press:

```text
q
```

The program needs to receive the keyboard event through the OpenCV window.

---

# 📁 Project Structure

```text
Virtual-steering-wheel/
│
├── main.py
├── hand_landmarker.task
├── README.md
├── LICENSE
└── .gitignore
```

---

# 🔧 Technologies Used

### Python

Main programming language.

### OpenCV

Used for:

* Webcam access
* Video processing
* Displaying the camera feed

### MediaPipe

Used for:

* Detecting hands
* Tracking hand landmarks

### PyDirectInput

Used to send keyboard input to the game.

---

# 🚀 Future Improvements

Possible future features include:

* Accelerator gesture
* Brake gesture
* Adjustable steering sensitivity
* Better hand detection
* Steering-wheel visualization
* Calibration system
* Support for more games
* Improved performance and reduced latency

---

# 📜 License

This project is open-source. See the `LICENSE` file for details.

---

# 👨‍💻 Author

**Samyak Jain**

Built as a computer-vision project to experiment with hand tracking and game control.
