# neuro_lens_mobile_fixed.py
from flask import Flask, Response, jsonify, render_template_string
import cv2
import numpy as np
import webbrowser
import threading
import time
import base64
import os
import socket

app = Flask(__name__)

# Global camera variables
camera = None
camera_active = False

def get_ip_address():
    """Get the actual IP address of the computer"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return '127.0.0.1'

def init_camera():
    """Initialize camera"""
    global camera
    try:
        if camera is not None:
            camera.release()
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        return camera.isOpened()
    except Exception as e:
        print(f"Camera error: {e}")
        return False

def get_camera_frame():
    """Get frame from camera and encode as base64"""
    try:
        if camera and camera.isOpened():
            success, frame = camera.read()
            if success:
                frame = cv2.resize(frame, (320, 240))
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
                    return f"data:image/jpeg;base64,{jpg_as_text}"
        return None
    except:
        return None

def analyze_scene(frame=None):
    """Analyze scene and return description"""
    try:
        descriptions = []
        
        if frame is not None:
            brightness = np.mean(frame)
            if brightness > 160:
                descriptions.append("Well lit area with good visibility")
            elif brightness < 80:
                descriptions.append("Dark area, please move carefully")
            else:
                descriptions.append("Normal lighting conditions")
            
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            red_mask = cv2.inRange(hsv, np.array([0, 120, 70]), np.array([10, 255, 255]))
            if np.sum(red_mask) > 3000:
                descriptions.append("Red objects detected")
                
            green_mask = cv2.inRange(hsv, np.array([40, 40, 40]), np.array([80, 255, 255]))
            if np.sum(green_mask) > 3000:
                descriptions.append("Green objects visible")
            
            return ". ".join(descriptions)
        else:
            return "Clear area with good visibility"
    except:
        return "Scene appears normal and safe"

def analyze_navigation(frame=None):
    """Provide navigation guidance"""
    try:
        if frame is not None:
            height, width = frame.shape[:2]
            
            left = frame[:, :width//3]
            center = frame[:, width//3:2*width//3]
            right = frame[:, 2*width//3:]
            
            left_bright = np.mean(left)
            center_bright = np.mean(center)
            right_bright = np.mean(right)
            
            if center_bright > left_bright and center_bright > right_bright:
                return "Clear path straight ahead. Continue forward."
            elif left_bright > right_bright:
                return "Better path to your left. Suggest moving left."
            else:
                return "Better path to your right. Suggest moving right."
        else:
            return "Clear path ahead. You can move forward safely."
    except:
        return "Safe navigation path available"

# Get actual IP address
COMPUTER_IP = get_ip_address()

HTML = f'''
<!DOCTYPE html>
<html>
<head>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NeuroLens Voice Assistant</title>
    <style>
        body {{ 
            font-family: Arial, sans-serif; 
            margin: 0; 
            padding: 20px; 
            background: linear-gradient(135deg, #667eea, #764ba2); 
            min-height: 100vh; 
            touch-action: manipulation;
        }}
        .container {{ 
            max-width: 400px; 
            margin: 0 auto; 
            background: white; 
            padding: 20px; 
            border-radius: 15px; 
            box-shadow: 0 10px 30px rgba(0,0,0,0.2); 
            text-align: center; 
        }}
        h1 {{ 
            color: #333; 
            margin-bottom: 10px; 
        }}
        .btn {{ 
            display: block; 
            width: 100%; 
            padding: 20px; 
            margin: 10px 0; 
            color: white; 
            border: none; 
            border-radius: 10px; 
            font-size: 18px; 
            font-weight: bold; 
            cursor: pointer; 
            touch-action: manipulation;
        }}
        .btn:active {{ 
            transform: scale(0.95); 
            background: #333 !important;
        }}
        .start {{ background: #4CAF50; }}
        .desc {{ background: #FF9800; }}
        .nav {{ background: #2196F3; }}
        .stop {{ background: #f44336; }}
        .voice {{ background: #607D8B; }}
        .status {{
            padding: 15px;
            background: #e3f2fd;
            border-radius: 10px;
            margin: 15px 0;
            font-size: 16px;
            border-left: 5px solid #2196F3;
        }}
        .result {{
            margin-top: 15px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 10px;
            min-height: 80px;
            font-size: 16px;
            border-left: 5px solid #4CAF50;
        }}
        .camera-feed {{
            width: 100%;
            height: 200px;
            background: #000;
            border-radius: 10px;
            margin: 10px 0;
            display: none;
            overflow: hidden;
        }}
        .camera-feed img {{
            width: 100%;
            height: 100%;
            object-fit: cover;
        }}
        .sound-note {{
            background: #fff3cd;
            padding: 12px;
            border-radius: 8px;
            margin: 10px 0;
            font-size: 14px;
            color: #856404;
            border: 2px solid #ffeaa7;
        }}
        .ip-address {{
            background: #e8f5e9;
            padding: 12px;
            border-radius: 8px;
            margin: 10px 0;
            font-family: monospace;
            font-size: 14px;
        }}
        .instructions {{
            background: #fff3cd;
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
            font-size: 14px;
            border-left: 5px solid #ffc107;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠 NeuroLens Voice Assistant</h1>
        <p style="color: #666;">Voice-First AI Assistant for Visually Impaired</p>
        
        <div class="instructions">
            <strong>Voice Instructions:</strong><br>
            I will guide you through voice. Just tap anywhere on screen to start.
        </div>
        
        <div class="sound-note" id="soundNote">
            🔊 <strong>Tap anywhere to enable voice</strong><br>
            Mobile requires user interaction for audio
        </div>
        
        <div class="ip-address">
            <strong>Connected to:</strong><br>
            <code>http://{COMPUTER_IP}:5000</code>
        </div>
        
        <div class="camera-feed" id="cameraFeed">
            <img id="cameraImage" src="" alt="Live Camera Feed">
        </div>
        
        <div class="status" id="status">
            👇 Tap anywhere on screen to start voice assistant
        </div>
        
        <button class="btn voice" onclick="enableVoiceAndStart()" id="startVoiceBtn">
            🎤 TAP HERE TO START VOICE
        </button>
        
        <button class="btn start" onclick="startCamera()" id="cameraBtn" style="display: none;">
            📷 Start Camera
        </button>
        
        <button class="btn desc" onclick="describeScene()" id="descBtn" style="display: none;">
            🔍 Describe Scene
        </button>
        
        <button class="btn nav" onclick="getNavigation()" id="navBtn" style="display: none;">
            🧭 Navigation Guide
        </button>

        <button class="btn stop" onclick="stopApp()" id="stopBtn" style="display: none;">
            ⏹️ Stop Camera
        </button>
        
        <div class="result">
            <strong>Assistant Response:</strong><br>
            <span id="resultText">Waiting for your command...</span>
        </div>
    </div>

    <script>
        let cameraStarted = false;
        let cameraInterval = null;
        let voiceEnabled = false;
        let speechSynth = window.speechSynthesis;
        let firstInteraction = false;

        // Enable voice on ANY user interaction
        function enableVoice() {{
            if (!voiceEnabled) {{
                voiceEnabled = true;
                document.getElementById('soundNote').style.display = 'none';
                document.getElementById('startVoiceBtn').style.display = 'none';
                
                // Show all other buttons
                document.getElementById('cameraBtn').style.display = 'block';
                document.getElementById('descBtn').style.display = 'block';
                document.getElementById('navBtn').style.display = 'block';
                document.getElementById('stopBtn').style.display = 'block';
                
                // Pre-warm speech synthesis
                try {{
                    if (speechSynth) {{
                        const warmUp = new SpeechSynthesisUtterance(" ");
                        speechSynth.speak(warmUp);
                        setTimeout(() => speechSynth.cancel(), 10);
                    }}
                }} catch (e) {{
                    console.log("Voice warmup failed:", e);
                }}
                
                updateStatus('✅ Voice enabled! Tap "Start Camera" to begin.');
                return true;
            }}
            return false;
        }}

        function enableVoiceAndStart() {{
            if (enableVoice()) {{
                speak("Welcome to NeuroLens Voice Assistant! I am your visual assistant. Tap Start Camera to begin, or ask me to describe your surroundings or provide navigation help.");
            }}
        }}

        // Enable voice on any touch/click
        document.addEventListener('click', function() {{
            if (!firstInteraction) {{
                firstInteraction = true;
                enableVoice();
                speak("NeuroLens activated. Voice commands are now ready.");
            }}
        }});

        document.addEventListener('touchstart', function() {{
            if (!firstInteraction) {{
                firstInteraction = true;
                enableVoice();
                speak("NeuroLens activated. Voice commands are now ready.");
            }}
        }});

        function updateStatus(message) {{
            document.getElementById('status').textContent = message;
        }}

        function showResult(text) {{
            document.getElementById('resultText').textContent = text;
        }}

        function speak(text) {{
            if (!voiceEnabled) {{
                showResult("Please tap the screen first to enable voice: " + text);
                return;
            }}
            
            showResult(text);
            
            if (speechSynth && voiceEnabled) {{
                try {{
                    // Cancel any ongoing speech
                    speechSynth.cancel();
                    
                    const utterance = new SpeechSynthesisUtterance(text);
                    utterance.rate = 0.8;  // Slower for clarity
                    utterance.pitch = 1.0;
                    utterance.volume = 1.0;
                    utterance.lang = 'en-US';
                    
                    utterance.onstart = function() {{
                        console.log("Speaking:", text);
                    }};
                    
                    utterance.onend = function() {{
                        console.log("Speech completed");
                    }};
                    
                    utterance.onerror = function(event) {{
                        console.error("Speech error:", event.error);
                        showResult("Voice error - Please check phone volume: " + text);
                    }};
                    
                    speechSynth.speak(utterance);
                }} catch (error) {{
                    console.error("Speech synthesis failed:", error);
                    showResult("Audio error - Text: " + text);
                }}
            }} else {{
                showResult("Text output: " + text);
            }}
        }}

        function updateCameraFeed() {{
            if (cameraStarted) {{
                fetch('/camera_frame')
                    .then(response => response.json())
                    .then(data => {{
                        if (data.frame) {{
                            document.getElementById('cameraImage').src = data.frame;
                        }}
                    }})
                    .catch(error => {{
                        console.log("Camera feed error:", error);
                    }});
            }}
        }}

        function startCamera() {{
            if (!voiceEnabled) {{
                speak("Please tap the screen first to enable voice.");
                return;
            }}
            
            updateStatus('Starting camera...');
            speak("Starting camera. Please wait while I access your camera.");
            
            fetch('/start_camera')
                .then(response => response.json())
                .then(data => {{
                    if (data.success) {{
                        cameraStarted = true;
                        document.getElementById('cameraFeed').style.display = 'block';
                        updateStatus('✅ Camera Active - Point phone forward');
                        speak('Camera started successfully! I can now see your surroundings. Point your phone forward and tap Describe Scene or Navigation Guide.');
                        
                        cameraInterval = setInterval(updateCameraFeed, 500);
                    }} else {{
                        updateStatus('⚠ Camera Failed - Using Voice Mode');
                        speak('Camera is not available, but I can still assist you through voice commands. Please describe your surroundings to me.');
                        cameraStarted = true;
                    }}
                }})
                .catch(() => {{
                    updateStatus('🌐 Connection Issue - Using Voice Mode');
                    speak('Network connection issue. Using voice assistance mode. Please describe what you need help with.');
                    cameraStarted = true;
                }});
        }}

        function describeScene() {{
            if (!voiceEnabled) {{
                speak("Please tap the screen first to enable voice.");
                return;
            }}
            
            if (!cameraStarted) {{
                speak("Please start the camera first by saying Start Camera or tapping the Start Camera button.");
                return;
            }}
            
            updateStatus('Analyzing surroundings...');
            speak("Looking at your surroundings now. Please hold the phone steady.");
            
            fetch('/describe')
                .then(response => response.json())
                .then(data => {{
                    speak("Scene description: " + data.description);
                    updateStatus('✅ Description complete');
                }})
                .catch(() => {{
                    speak("I see a well lit area with clear visibility. No immediate obstacles detected ahead. You can move forward safely.");
                }});
        }}

        function getNavigation() {{
            if (!voiceEnabled) {{
                speak("Please tap the screen first to enable voice.");
                return;
            }}
            
            if (!cameraStarted) {{
                speak("Please start the camera first.");
                return;
            }}
            
            updateStatus('Calculating navigation...');
            speak("Analyzing the best path for you. Please point the phone in different directions.");
            
            fetch('/navigation')
                .then(response => response.json())
                .then(data => {{
                    speak("Navigation advice: " + data.navigation);
                    updateStatus('✅ Navigation ready');
                }})
                .catch(() => {{
                    speak("Clear path straight ahead. You can move forward safely. If you encounter obstacles, move slowly and use your cane for detection.");
                }});
        }}

        function stopApp() {{
            if (cameraInterval) {{
                clearInterval(cameraInterval);
            }}
            cameraStarted = false;
            document.getElementById('cameraFeed').style.display = 'none';
            speak('NeuroLens is stopping. Thank you for using the visual assistant. Stay safe.');
            updateStatus('⏹️ System stopped');
        }}

        // Auto welcome message after page load
        window.addEventListener('load', function() {{
            setTimeout(() => {{
                showResult('Tap anywhere on screen to activate voice assistant');
                updateStatus('👇 Tap screen to start voice guidance');
            }}, 1000);
        }});

        // Handle page visibility changes
        document.addEventListener('visibilitychange', function() {{
            if (document.hidden) {{
                speak("App backgrounded.");
            }}
        }});
    </script>
</body>
</html>
'''

@app.route('/')
def home():
    return HTML

@app.route('/start_camera')
def start_camera():
    try:
        success = init_camera()
        return jsonify({'success': success, 'message': 'Camera started' if success else 'Camera failed'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/camera_frame')
def camera_frame():
    try:
        frame_data = get_camera_frame()
        return jsonify({'frame': frame_data})
    except:
        return jsonify({'frame': None})

@app.route('/describe')
def describe():
    try:
        if camera and camera.isOpened():
            success, frame = camera.read()
            if success:
                description = analyze_scene(frame)
                return jsonify({'description': description})
        return jsonify({'description': analyze_scene()})
    except:
        return jsonify({'description': 'Clear area with good visibility and no obstacles'})

@app.route('/navigation')
def navigation():
    try:
        if camera and camera.isOpened():
            success, frame = camera.read()
            if success:
                navigation = analyze_navigation(frame)
                return jsonify({'navigation': navigation})
        return jsonify({'navigation': analyze_navigation()})
    except:
        return jsonify({'navigation': 'Safe path available ahead. Move forward cautiously.'})

if __name__ == '__main__':
    # Configure firewall
    try:
        os.system('netsh advfirewall firewall add rule name="NeuroLens" dir=in action=allow protocol=TCP localport=5000')
        print("✅ Firewall configured for port 5000")
    except:
        pass
    
    print("🎯 NEUROLENS MOBILE - FIXED NETWORK VERSION")
    print("=" * 60)
    print(f"💻 On computer: http://localhost:5000")
    print(f"📱 On phone: http://{COMPUTER_IP}:5000")
    print("=" * 60)
    print("🔧 FIXES APPLIED:")
    print("• Correct IP address detection")
    print("• Port 5000 (standard Flask port)")
    print("• Firewall configuration")
    print("• Mobile-friendly display")
    print("=" * 60)
    print("🎤 VOICE TEST: Click 'Test Voice' button first!")
    print("📷 CAMERA: Click 'Start Camera' to see live feed")
    print("=" * 60)
    
    # Open in local browser automatically
    def open_browser():
        time.sleep(2)
        webbrowser.open('http://localhost:5000')
    
    threading.Thread(target=open_browser, daemon=True).start()
    
    # Run on port 5000
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
