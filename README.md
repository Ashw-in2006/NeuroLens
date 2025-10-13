# NeuroLens — Enhanced Navigation for the Visually Impaired

NeuroLens is a real-time navigation assistant that uses your device’s camera to detect obstacles, assess path safety, and provide voice-guided directions. It is meant to help visually impaired users move more safely in their environment.

---

## 🧩 Requirements

- Python 3.8 or newer  
- A camera (webcam or phone camera via browser)  
- Browser with microphone / speech synthesis support  

### Python Dependencies

Install via pip:

```bash
pip install flask opencv-python numpy
▶️ Running the App
Run the main Python file:

bash
Copy code
python neuro_lens_perfect_navigation_complete.py
You will see output like:

markdown
Copy code
🎯 NEUROLENS ENHANCED NAVIGATION - COMPLETE FIX
============================================================
💻 On computer: http://localhost:5000
📱 On phone: http://192.168.x.x:5000
============================================================
💻 Access Instructions
From PC
Open your browser and go to:

arduino
Copy code
http://localhost:5000
From Mobile (same Wi-Fi network)
Make sure your phone and PC are on the same Wi-Fi network.

In the terminal output, note the “On phone” address (for example http://192.168.x.x:5000).

Open that address in your mobile browser.

Tap “START NAVIGATION” to begin.

🎤 Voice Guidance & Controls
You must tap the screen once (or interact) to enable voice output (required by browsers).

Once enabled, NeuroLens will speak warnings and navigation instructions (e.g. “Obstacle ahead on the right,” “Turn left,” etc.).

Buttons allow starting/stopping navigation and requesting guidance.

🔧 Troubleshooting
Problem	Solution
Camera fails to start	Check if another app is using the camera, or allow camera permission
No voice / sound	Tap screen to enable voice; check browser audio settings
Mobile unable to connect	Ensure both devices are on same network, and firewall isn’t blocking port 5000
“Camera failed” error	Restart the Python script or your computer; confirm OpenCV can access your camera

📂 Repository Structure
Currently, the project contains:

Copy code
├── neuro_lens_perfect_navigation_complete.py
└── README.md
📝 License & Credits
NeuroLens is open source. Feel free to use, modify, and share.
Credits: Developed to help with safe navigation and accessibility improvements.

Thank you for using NeuroLens!
Stay safe, stay independent.

pgsql
Copy code

If you like, I can also generate a **markdown preview screenshot** and include badges (build status, license) that you can drop into the README. Do you want me to add those?
::contentReference[oaicite:0]{index=0}
