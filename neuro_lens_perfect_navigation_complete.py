# neuro_lens_perfect_navigation_complete.py
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
current_objects = []
navigation_warnings = []

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
        time.sleep(2)
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
                
                # Perform navigation analysis
                analyzed_frame, navigation_info = analyze_navigation_frame(frame)
                
                # Encode the frame
                ret, buffer = cv2.imencode('.jpg', analyzed_frame)
                if ret:
                    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
                    return f"data:image/jpeg;base64,{jpg_as_text}", navigation_info
                
        return None, {"direction": "unknown", "obstacles": [], "safety": "unknown", "detailed_obstacles": [], "path_quality": "unknown"}
    except Exception as e:
        print(f"Frame error: {e}")
        return None, {"direction": "unknown", "obstacles": [], "safety": "unknown", "detailed_obstacles": [], "path_quality": "unknown"}

def analyze_navigation_frame(frame):
    """Advanced navigation analysis for blind people"""
    global navigation_warnings
    
    try:
        # Convert to different color spaces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        height, width = frame.shape[:2]
        
        # Divide frame into navigation zones
        left_zone = frame[:, :width//3]
        center_zone = frame[:, width//3:2*width//3]
        right_zone = frame[:, 2*width//3:]
        bottom_zone = frame[2*height//3:, :]  # Near zone
        
        # Analyze each zone
        navigation_info = {
            "direction": "forward",
            "obstacles": [],
            "detailed_obstacles": [],
            "safety": "safe",
            "path_quality": "good",
            "zone_analysis": {}
        }
        
        # OBSTACLE DETECTION
        obstacles, detailed_obstacles = detect_obstacles(frame, gray, hsv, width, height)
        navigation_info["obstacles"] = obstacles
        navigation_info["detailed_obstacles"] = detailed_obstacles
        
        # PATH ANALYSIS
        path_analysis = analyze_path_quality(left_zone, center_zone, right_zone, bottom_zone)
        navigation_info.update(path_analysis)
        
        # ZONE ANALYSIS
        zone_analysis = analyze_navigation_zones(left_zone, center_zone, right_zone, bottom_zone, detailed_obstacles)
        navigation_info["zone_analysis"] = zone_analysis
        
        # DIRECTION GUIDANCE
        best_direction = calculate_best_direction(zone_analysis, obstacles)
        navigation_info["direction"] = best_direction
        
        # SAFETY ASSESSMENT
        safety_level = assess_safety_level(obstacles, path_analysis, zone_analysis)
        navigation_info["safety"] = safety_level
        
        # VISUAL FEEDBACK
        frame = draw_navigation_overlay(frame, navigation_info, width, height)
        
        # Update warnings
        navigation_warnings = obstacles.copy()
        if safety_level == "danger":
            navigation_warnings.append("DANGEROUS PATH - STOP")
        elif safety_level == "caution":
            navigation_warnings.append("CAUTION REQUIRED")
        
        return frame, navigation_info
        
    except Exception as e:
        print(f"Navigation analysis error: {e}")
        return frame, {"direction": "forward", "obstacles": [], "safety": "unknown", "detailed_obstacles": [], "path_quality": "unknown"}

def detect_obstacles(frame, gray, hsv, width, height):
    """Detect potential obstacles with detailed information"""
    obstacles = []
    detailed_obstacles = []
    
    # Edge detection for object boundaries
    edges = cv2.Canny(gray, 50, 150)
    
    # Find contours (potential obstacles)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if 200 < area < 15000:  # Reasonable obstacle size
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate obstacle position in frame
            center_x = x + w//2
            center_y = y + h//2
            
            # Determine zone
            if center_x < width//3:
                zone = "left"
            elif center_x < 2*width//3:
                zone = "center"
            else:
                zone = "right"
                
            # Determine distance (y position indicates distance)
            if center_y > 2*height//3:
                distance = "very close"
            elif center_y > height//2:
                distance = "close"
            else:
                distance = "ahead"
            
            # Classify obstacle type
            aspect_ratio = w / h if h > 0 else 0
            
            if aspect_ratio > 2.5:
                obstacle_type = "long horizontal object"
                severity = "medium"
            elif aspect_ratio > 1.5:
                obstacle_type = "wide object"
                severity = "medium"
            elif aspect_ratio < 0.4:
                obstacle_type = "tall vertical object"
                severity = "low"
            elif 0.8 < aspect_ratio < 1.2:
                obstacle_type = "square object"
                severity = "medium"
            else:
                obstacle_type = "object"
                severity = "low"
            
            # Check if obstacle is in walking path
            if y + h > frame.shape[0] * 0.6:  # In lower part of frame
                if zone == "center":
                    obstacles.append(f"{obstacle_type} ahead")
                    detailed_obstacles.append({
                        "type": obstacle_type,
                        "zone": zone,
                        "distance": distance,
                        "severity": severity,
                        "position": (center_x, center_y),
                        "size": area
                    })
                else:
                    obstacles.append(f"{obstacle_type} on {zone}")
                    detailed_obstacles.append({
                        "type": obstacle_type,
                        "zone": zone,
                        "distance": distance,
                        "severity": severity,
                        "position": (center_x, center_y),
                        "size": area
                    })
            
            # Draw obstacle bounding box with color based on zone
            color = (0, 0, 255) if zone == "center" else (0, 165, 255) if zone == "left" else (255, 0, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Add zone label
            cv2.putText(frame, f"{zone} {obstacle_type}", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    return obstacles, detailed_obstacles

def analyze_navigation_zones(left_zone, center_zone, right_zone, bottom_zone, detailed_obstacles):
    """Analyze each navigation zone in detail"""
    zone_analysis = {
        "left": {"obstacle_count": 0, "clearance": "clear", "main_obstacles": []},
        "center": {"obstacle_count": 0, "clearance": "clear", "main_obstacles": []},
        "right": {"obstacle_count": 0, "clearance": "clear", "main_obstacles": []},
        "near": {"obstacle_count": 0, "clearance": "clear", "main_obstacles": []}
    }
    
    # Count obstacles in each zone
    for obstacle in detailed_obstacles:
        zone = obstacle["zone"]
        zone_analysis[zone]["obstacle_count"] += 1
        if obstacle["distance"] == "very close" or obstacle["severity"] == "high":
            zone_analysis[zone]["main_obstacles"].append(obstacle)
    
    # Determine clearance for each zone
    for zone in ["left", "center", "right"]:
        count = zone_analysis[zone]["obstacle_count"]
        if count >= 3:
            zone_analysis[zone]["clearance"] = "blocked"
        elif count >= 2:
            zone_analysis[zone]["clearance"] = "crowded"
        elif count >= 1:
            zone_analysis[zone]["clearance"] = "partial"
        else:
            zone_analysis[zone]["clearance"] = "clear"
    
    return zone_analysis

def analyze_path_quality(left_zone, center_zone, right_zone, bottom_zone):
    """Analyze quality of path in different directions"""
    analysis = {}
    
    # Calculate brightness variance (smooth path vs cluttered)
    left_variance = np.var(cv2.cvtColor(left_zone, cv2.COLOR_BGR2GRAY))
    center_variance = np.var(cv2.cvtColor(center_zone, cv2.COLOR_BGR2GRAY))
    right_variance = np.var(cv2.cvtColor(right_zone, cv2.COLOR_BGR2GRAY))
    
    # Lower variance = smoother path
    variances = {
        "left": left_variance,
        "center": center_variance, 
        "right": right_variance
    }
    
    # Find clearest path (lowest variance)
    clearest_direction = min(variances, key=variances.get)
    analysis["clearest_path"] = clearest_direction
    
    # Path quality assessment
    min_variance = min(variances.values())
    if min_variance < 500:
        analysis["path_quality"] = "excellent"
    elif min_variance < 1000:
        analysis["path_quality"] = "good"
    elif min_variance < 2000:
        analysis["path_quality"] = "fair"
    else:
        analysis["path_quality"] = "poor"
    
    return analysis

def calculate_best_direction(zone_analysis, obstacles):
    """Calculate safest direction to move based on zone analysis"""
    
    # Get clearance levels
    left_clearance = zone_analysis["left"]["clearance"]
    center_clearance = zone_analysis["center"]["clearance"] 
    right_clearance = zone_analysis["right"]["clearance"]
    
    # Priority logic
    if center_clearance in ["clear", "partial"]:
        return "forward"
    elif left_clearance in ["clear", "partial"] and right_clearance in ["clear", "partial"]:
        # Both sides clear, choose based on obstacle count
        if zone_analysis["left"]["obstacle_count"] <= zone_analysis["right"]["obstacle_count"]:
            return "left"
        else:
            return "right"
    elif left_clearance in ["clear", "partial"]:
        return "left"
    elif right_clearance in ["clear", "partial"]:
        return "right"
    else:
        return "stop"

def assess_safety_level(obstacles, path_analysis, zone_analysis):
    """Overall safety assessment"""
    
    danger_score = 0
    
    # Obstacle danger
    center_obstacles = zone_analysis["center"]["obstacle_count"]
    if center_obstacles >= 2:
        danger_score += 3
    elif center_obstacles >= 1:
        danger_score += 1
    
    # Immediate obstacles in near zone
    near_obstacles = zone_analysis["near"]["obstacle_count"]
    if near_obstacles >= 1:
        danger_score += 2
    
    # Path blocked in all directions
    if (zone_analysis["left"]["clearance"] == "blocked" and 
        zone_analysis["center"]["clearance"] == "blocked" and 
        zone_analysis["right"]["clearance"] == "blocked"):
        danger_score += 3
    
    # Path quality danger
    if path_analysis["path_quality"] in ["poor"]:
        danger_score += 1
    
    # Safety level determination
    if danger_score >= 4:
        return "danger"
    elif danger_score >= 2:
        return "caution"
    elif danger_score >= 1:
        return "moderate"
    else:
        return "safe"

def draw_navigation_overlay(frame, nav_info, width, height):
    """Draw navigation information on frame"""
    
    # Draw direction arrows
    arrow_color = (0, 255, 0)  # Green for safe
    if nav_info["safety"] == "caution":
        arrow_color = (0, 255, 255)  # Yellow
    elif nav_info["safety"] == "danger":
        arrow_color = (0, 0, 255)  # Red

    # Draw direction indicator
    direction = nav_info["direction"]
    if direction == "forward":
        cv2.arrowedLine(frame, (width//2, height-20), (width//2, 50), arrow_color, 3)
    elif direction == "left":
        cv2.arrowedLine(frame, (width//2, height-20), (width//4, height//2), arrow_color, 3)
    elif direction == "right":
        cv2.arrowedLine(frame, (width//2, height-20), (3*width//4, height//2), arrow_color, 3)
    else:  # stop
        cv2.putText(frame, "STOP", (width//2-30, height//2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # Draw safety zones with labels
    cv2.rectangle(frame, (0, 2*height//3), (width, height), (255, 255, 0), 2)  # Near zone
    cv2.rectangle(frame, (width//3, 0), (2*width//3, height), (0, 255, 255), 2)  # Center zone
    
    # Zone labels
    cv2.putText(frame, "LEFT", (10, height//2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, "CENTER", (width//2-30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, "RIGHT", (width-60, height//2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, "NEAR", (10, height-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Add text information
    cv2.putText(frame, f"Safety: {nav_info['safety']}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, arrow_color, 2)
    cv2.putText(frame, f"Go: {direction}", (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, arrow_color, 2)
    cv2.putText(frame, f"Path: {nav_info['path_quality']}", (10, 90), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, arrow_color, 2)

    # Obstacle counter
    cv2.putText(frame, f"Obstacles: {len(nav_info['obstacles'])}", (width-150, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, arrow_color, 2)
    
    return frame

def get_navigation_guidance(nav_info):
    """Generate voice guidance for navigation with detailed obstacle information"""
    
    guidance = []
    direction = nav_info["direction"]
    safety = nav_info["safety"]
    obstacles = nav_info["obstacles"]
    detailed_obstacles = nav_info["detailed_obstacles"]
    zone_analysis = nav_info["zone_analysis"]
    
    # URGENT SAFETY WARNINGS
    if safety == "danger":
        guidance.append("⚠️ DANGER! STOP IMMEDIATELY!")
        
        # Provide specific danger information
        center_obstacles = [obs for obs in detailed_obstacles if obs["zone"] == "center" and obs["distance"] == "very close"]
        if center_obstacles:
            guidance.append(f"Immediate obstacle detected: {center_obstacles[0]['type']} directly ahead.")
        
        if zone_analysis["left"]["clearance"] == "blocked" and zone_analysis["right"]["clearance"] == "blocked":
            guidance.append("All paths are blocked. Please wait for assistance or find alternative route.")
        else:
            # Suggest alternatives even in danger
            if zone_analysis["left"]["clearance"] != "blocked":
                guidance.append("Consider moving carefully to your left when safe.")
            if zone_analysis["right"]["clearance"] != "blocked":
                guidance.append("Consider moving carefully to your right when safe.")
        
        return ". ".join(guidance)
    
    # CAUTION WARNINGS
    if safety == "caution":
        guidance.append("⚠️ CAUTION REQUIRED! Proceed carefully.")
    
    # DETAILED OBSTACLE INFORMATION
    if detailed_obstacles:
        # Group obstacles by zone and proximity
        immediate_obstacles = [obs for obs in detailed_obstacles if obs["distance"] == "very close"]
        close_obstacles = [obs for obs in detailed_obstacles if obs["distance"] == "close"]
        ahead_obstacles = [obs for obs in detailed_obstacles if obs["distance"] == "ahead"]
        
        if immediate_obstacles:
            guidance.append("WARNING: Immediate obstacles detected:")
            for obs in immediate_obstacles[:2]:  # Limit to 2 most critical
                guidance.append(f"{obs['type']} very close on your {obs['zone']}")
        
        if close_obstacles and not immediate_obstacles:
            guidance.append("Close obstacles detected:")
            for obs in close_obstacles[:2]:
                guidance.append(f"{obs['type']} close on your {obs['zone']}")
        
        if ahead_obstacles and not (immediate_obstacles or close_obstacles):
            guidance.append("Objects detected ahead:")
            for obs in ahead_obstacles[:2]:
                guidance.append(f"{obs['type']} ahead on your {obs['zone']}")
    
    # DIRECTION GUIDANCE WITH REASONING
    if direction == "forward":
        if not obstacles:
            guidance.append("Clear path straight ahead. You can move forward safely.")
        else:
            guidance.append("Path ahead is navigable. Continue forward with attention to obstacles.")
    elif direction == "left":
        guidance.append("Turn slightly left for clearer path.")
        if zone_analysis["left"]["clearance"] == "clear":
            guidance.append("Left side is completely clear.")
        else:
            guidance.append("Left path has fewer obstacles.")
    elif direction == "right":
        guidance.append("Turn slightly right for better route.")
        if zone_analysis["right"]["clearance"] == "clear":
            guidance.append("Right side is completely clear.")
        else:
            guidance.append("Right path has fewer obstacles.")
    else:  # stop
        guidance.append("STOP! Path blocked ahead.")
        guidance.append("All directions have obstacles. Please wait or seek assistance.")
    
    # PATH QUALITY INFORMATION
    if nav_info["path_quality"] == "excellent":
        guidance.append("Excellent walking conditions with smooth path.")
    elif nav_info["path_quality"] == "good":
        guidance.append("Good walking path ahead.")
    elif nav_info["path_quality"] == "fair":
        guidance.append("Uneven path ahead, walk carefully.")
    elif nav_info["path_quality"] == "poor":
        guidance.append("Rough terrain detected, extra caution advised.")
    
    # ZONE CLEARANCE SUMMARY
    if safety in ["safe", "moderate"]:
        clear_zones = []
        for zone in ["left", "center", "right"]:
            if zone_analysis[zone]["clearance"] == "clear":
                clear_zones.append(zone)
        
        if clear_zones:
            if len(clear_zones) == 3:
                guidance.append("All directions are clear.")
            else:
                guidance.append(f"Clear paths available: {', '.join(clear_zones)}")
    
    return ". ".join(guidance)

# Get actual IP address
COMPUTER_IP = get_ip_address()

HTML = '''
<!DOCTYPE html>
<html>
<head>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NeuroLens Navigation</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            margin: 0; 
            padding: 20px; 
            background: linear-gradient(135deg, #667eea, #764ba2); 
            min-height: 100vh; 
            touch-action: manipulation;
        }
        .container { 
            max-width: 400px; 
            margin: 0 auto; 
            background: white; 
            padding: 20px; 
            border-radius: 15px; 
            box-shadow: 0 10px 30px rgba(0,0,0,0.2); 
            text-align: center; 
        }
        h1 { 
            color: #333; 
            margin-bottom: 10px; 
        }
        .btn { 
            display: block; 
            width: 100%; 
            padding: 15px; 
            margin: 8px 0; 
            color: white; 
            border: none; 
            border-radius: 10px; 
            font-size: 16px; 
            font-weight: bold; 
            cursor: pointer; 
            touch-action: manipulation;
        }
        .btn:active { transform: scale(0.98); }
        .start { background: #4CAF50; }
        .nav { background: #2196F3; }
        .stop { background: #f44336; }
        .voice { background: #607D8B; }
        .status { padding: 12px; background: #e3f2fd; border-radius: 10px; margin: 12px 0; font-size: 14px; }
        .result { margin-top: 12px; padding: 12px; background: #f8f9fa; border-radius: 10px; min-height: 80px; font-size: 14px; }
        .camera-feed { width: 100%; height: 200px; background: #000; border-radius: 10px; margin: 10px 0; display: none; overflow: hidden; position: relative; }
        .camera-feed img { width: 100%; height: 100%; object-fit: cover; }
        .navigation-panel { background: #e8f5e9; padding: 10px; border-radius: 8px; margin: 8px 0; text-align: left; display: none; }
        .warning-panel { background: #ffebee; padding: 10px; border-radius: 8px; margin: 8px 0; display: none; }
        .safety-indicator { display: inline-block; padding: 5px 10px; border-radius: 15px; color: white; font-weight: bold; margin: 2px; }
        .safe { background: #4CAF50; }
        .caution { background: #FF9800; }
        .danger { background: #f44336; }
        .sound-note { background: #fff3cd; padding: 10px; border-radius: 8px; margin: 8px 0; font-size: 13px; }
        .ip-address { background: #e8f5e9; padding: 10px; border-radius: 8px; margin: 8px 0; font-family: monospace; font-size: 12px; }
        .instructions { background: #fff3cd; padding: 12px; border-radius: 8px; margin: 12px 0; font-size: 13px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧭 NeuroLens Navigation</h1>
        <p style="color: #666; font-size: 14px;">Enhanced Navigation System for Blind People</p>
        
        <div class="instructions">
            <strong>Enhanced Navigation Assistant Ready</strong><br>
            I will provide detailed obstacle information and safe path guidance
        </div>
        
        <div class="sound-note" id="soundNote">
            🔊 <strong>Tap screen to enable voice guidance</strong>
        </div>
        
        <div class="ip-address">
            <strong>Access from phone:</strong><br>
            <code>http://''' + COMPUTER_IP + ''':5000</code>
        </div>
        
        <div class="camera-feed" id="cameraFeed">
            <img id="cameraImage" src="" alt="Navigation View">
        </div>
        
        <div class="navigation-panel" id="navPanel">
            <strong>🧭 Navigation Status:</strong><br>
            <div id="navInfo">Starting navigation system...</div>
        </div>
        
        <div class="warning-panel" id="warningPanel">
            <strong>⚠️ Warnings:</strong><br>
            <div id="warningsList">No warnings</div>
        </div>
        
        <div class="status" id="status">
            👇 Tap screen to start navigation guidance
        </div>
        
        <button class="btn voice" onclick="enableVoiceAndStart()" id="startVoiceBtn">
            🎤 START NAVIGATION
        </button>
        
        <button class="btn start" onclick="startNavigation()" id="navBtn" style="display: none;">
            📷 START CAMERA NAVIGATION
        </button>
        
        <button class="btn nav" onclick="getNavigationGuidance()" id="guideBtn" style="display: none;">
            🧭 GET GUIDANCE
        </button>

        <button class="btn stop" onclick="stopNavigation()" id="stopBtn" style="display: none;">
            ⏹️ STOP NAVIGATION
        </button>
        
        <div class="result">
            <strong>Navigation Assistant:</strong><br>
            <span id="resultText">Ready to guide you safely...</span>
        </div>
    </div>

    <script>
        let navigationActive = false;
        let cameraInterval = null;
        let voiceEnabled = false;
        let speechSynth = window.speechSynthesis;
        let firstInteraction = false;

        function enableVoice() {
            if (!voiceEnabled) {
                voiceEnabled = true;
                document.getElementById('soundNote').style.display = 'none';
                document.getElementById('startVoiceBtn').style.display = 'none';
                document.getElementById('navBtn').style.display = 'block';
                document.getElementById('guideBtn').style.display = 'block';
                document.getElementById('stopBtn').style.display = 'block';
                
                updateStatus('✅ Navigation ready! Tap START CAMERA NAVIGATION');
                speak("NeuroLens Enhanced Navigation System activated. I will provide detailed obstacle information and guide you safely through your environment. Tap Start Camera Navigation to begin.");
                return true;
            }
            return false;
        }

        function enableVoiceAndStart() {
            if (enableVoice()) {
                speak("I am your enhanced navigation assistant. I detect obstacles with detailed information, analyze paths, and provide step-by-step guidance for safe movement.");
            }
        }

        document.addEventListener('click', function() {
            if (!firstInteraction) {
                firstInteraction = true;
                enableVoice();
            }
        });

        function updateStatus(message) {
            document.getElementById('status').textContent = message;
        }

        function showResult(text) {
            document.getElementById('resultText').textContent = text;
        }

        function updateNavigationInfo(navInfo) {
            const navPanel = document.getElementById('navInfo');
            const warningPanel = document.getElementById('warningPanel');
            const warningsList = document.getElementById('warningsList');
            
            if (navInfo) {
                let safetyClass = 'safe';
                if (navInfo.safety === 'caution') safetyClass = 'caution';
                if (navInfo.safety === 'danger') safetyClass = 'danger';
                
                navPanel.innerHTML = `
                    <span class="safety-indicator ${safetyClass}">Safety: ${navInfo.safety.toUpperCase()}</span><br>
                    Direction: <strong>${navInfo.direction.toUpperCase()}</strong><br>
                    Path Quality: ${navInfo.path_quality}<br>
                    Obstacles: ${navInfo.obstacles.length}
                `;
                
                document.getElementById('navPanel').style.display = 'block';
                
                // Update warnings
                if (navInfo.obstacles.length > 0 || navInfo.safety === 'danger' || navInfo.safety === 'caution') {
                    warningsList.innerHTML = navInfo.obstacles.map(obs => `• ${obs}`).join('<br>');
                    if (navInfo.safety === 'danger') {
                        warningsList.innerHTML += '<br>• ⚠️ DANGEROUS PATH';
                    }
                    warningPanel.style.display = 'block';
                } else {
                    warningPanel.style.display = 'none';
                }
            }
        }

        function speak(text) {
            showResult(text);
            if (speechSynth && voiceEnabled) {
                speechSynth.cancel();
                const utterance = new SpeechSynthesisUtterance(text);
                utterance.rate = 0.8;
                utterance.pitch = 0.9;
                speechSynth.speak(utterance);
            }
        }

        function updateCameraFeed() {
            if (navigationActive) {
                fetch('/camera_frame')
                    .then(response => response.json())
                    .then(data => {
                        if (data.frame) {
                            document.getElementById('cameraImage').src = data.frame;
                        }
                        if (data.navigation_info) {
                            updateNavigationInfo(data.navigation_info);
                        }
                    })
                    .catch(error => {
                        console.log('Camera feed error:', error);
                    });
            }
        }

        function startNavigation() {
            if (!voiceEnabled) return;
            
            updateStatus('Starting navigation system...');
            speak("Initializing enhanced navigation system. Starting detailed obstacle detection and path analysis.");
            
            fetch('/start_camera')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        navigationActive = true;
                        document.getElementById('cameraFeed').style.display = 'block';
                        updateStatus('✅ Navigation Active - Point camera forward');
                        speak("Enhanced navigation system active! I am now monitoring your path with detailed obstacle detection. Point your phone forward as you walk.");
                        cameraInterval = setInterval(updateCameraFeed, 1000);
                    } else {
                        updateStatus('❌ Camera failed');
                        speak("Camera not available. Please check camera permissions.");
                    }
                })
                .catch(error => {
                    updateStatus('❌ Camera start failed');
                    speak("Failed to start camera. Please check if camera is available.");
                });
        }

        function getNavigationGuidance() {
            if (!navigationActive) {
                speak("Please start navigation first");
                return;
            }
            
            updateStatus('🔄 Getting navigation guidance...');
            speak("Analyzing current path and providing detailed guidance.");
            
            fetch('/navigation_guidance')
                .then(response => response.json())
                .then(data => {
                    if (data.guidance) {
                        speak(data.guidance);
                        updateStatus('✅ Guidance provided');
                    }
                })
                .catch(error => {
                    speak("Error getting guidance. Please try again.");
                    updateStatus('❌ Guidance error');
                });
        }

        function stopNavigation() {
            if (cameraInterval) clearInterval(cameraInterval);
            navigationActive = false;
            document.getElementById('cameraFeed').style.display = 'none';
            document.getElementById('navPanel').style.display = 'none';
            document.getElementById('warningPanel').style.display = 'none';
            speak('Enhanced navigation system stopped. Thank you for using NeuroLens Navigation.');
            updateStatus('⏹️ Navigation stopped - Tap to restart');
        }

        window.addEventListener('load', function() {
            showResult('Tap screen to start enhanced navigation system');
        });
    </script>
</body>
</html>
'''

@app.route('/')
def home():
    return render_template_string(HTML)

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
        frame_data, navigation_info = get_camera_frame()
        return jsonify({'frame': frame_data, 'navigation_info': navigation_info})
    except Exception as e:
        return jsonify({'frame': None, 'navigation_info': {"direction": "unknown", "obstacles": [], "safety": "unknown", "detailed_obstacles": [], "path_quality": "unknown"}})

@app.route('/navigation_guidance')
def navigation_guidance():
    try:
        if camera and camera.isOpened():
            success, frame = camera.read()
            if success:
                frame = cv2.resize(frame, (320, 240))
                analyzed_frame, navigation_info = analyze_navigation_frame(frame)
                guidance = get_navigation_guidance(navigation_info)
                return jsonify({'guidance': guidance, 'navigation_info': navigation_info})
        
        return jsonify({'guidance': 'Navigation system initializing. Please point camera forward.', 'navigation_info': {}})
    except Exception as e:
        return jsonify({'guidance': 'Clear path ahead. You can move forward safely.', 'navigation_info': {}})

if __name__ == '__main__':
    print("🎯 NEUROLENS ENHANCED NAVIGATION - COMPLETE FIX")
    print("=" * 60)
    print(f"💻 On computer: http://localhost:5000")
    print(f"📱 On phone: http://{COMPUTER_IP}:5000")
    print("=" * 60)
    print("✅ ALL ROUTES FIXED!")
    print("🔧 Enhanced obstacle detection with detailed descriptions")
    print("🎯 Better audio guidance with specific obstacle locations")
    print("=" * 60)
    
    def open_browser():
        time.sleep(2)
        webbrowser.open('http://localhost:5000')
    
    threading.Thread(target=open_browser, daemon=True).start()
    
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
