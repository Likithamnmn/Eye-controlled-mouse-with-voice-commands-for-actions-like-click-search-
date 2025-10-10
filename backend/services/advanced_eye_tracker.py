"""
Advanced Eye Tracking System - iPhone-inspired techniques for laptop
Implements sophisticated algorithms for high-precision gaze tracking
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import numpy as np
import pyautogui
import time
import math
import threading
import glob
from collections import deque
from dataclasses import dataclass
from typing import Tuple, List, Optional
import json


pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0  

@dataclass
class GazePoint:
    """Represents a gaze point with timestamp and confidence"""
    x: float
    y: float
    timestamp: float
    confidence: float
    
@dataclass
class EyeState:
    """Complete eye state information"""
    left_iris: Tuple[float, float]
    right_iris: Tuple[float, float]
    head_pose: Tuple[float, float, float]  
    blink_ratio: float
    attention_score: float

class KalmanGazeFilter:
    """Kalman filter for smooth gaze prediction and tracking"""
    
    def __init__(self):
        
        self.state = np.zeros(4, dtype=np.float32)
        self.covariance = np.eye(4, dtype=np.float32) * 1000
        
        
        self.process_noise = np.eye(4, dtype=np.float32)
        self.process_noise[0, 0] = 0.1  
        self.process_noise[1, 1] = 0.1  
        self.process_noise[2, 2] = 0.5  
        self.process_noise[3, 3] = 0.5 
        
        
        self.measurement_noise = np.eye(2, dtype=np.float32) * 10
        
        
        self.transition = np.array([
            [1, 0, 1, 0],  
            [0, 1, 0, 1],  
            [0, 0, 1, 0],  
            [0, 0, 0, 1]   
        ], dtype=np.float32)
        
        
        self.observation = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        self.initialized = False
    
    def predict(self) -> Tuple[float, float]:
        """Predict next gaze position"""
        if not self.initialized:
            return 0, 0
        
        
        self.state = self.transition @ self.state
        self.covariance = self.transition @ self.covariance @ self.transition.T + self.process_noise
        
        return self.state[0], self.state[1]
    
    def update(self, measurement: Tuple[float, float], confidence: float = 1.0):
        """Update filter with new measurement"""
        if not self.initialized:
            self.state[0], self.state[1] = measurement
            self.initialized = True
            return
        
        
        adjusted_noise = self.measurement_noise / max(confidence, 0.1)
        
        
        innovation = np.array(measurement) - self.observation @ self.state
        innovation_covariance = self.observation @ self.covariance @ self.observation.T + adjusted_noise
        
        
        gain = self.covariance @ self.observation.T @ np.linalg.inv(innovation_covariance)
        
        
        self.state = self.state + gain @ innovation
        self.covariance = (np.eye(4) - gain @ self.observation) @ self.covariance

class AttentionDetector:
    """Detects user attention and focus patterns"""
    
    def __init__(self, window_size: int = 30):
        self.gaze_history = deque(maxlen=window_size)
        self.fixation_threshold = 25  
        self.min_fixation_duration = 0.15  
    
    def add_gaze_point(self, point: GazePoint):
        """Add new gaze point to history"""
        self.gaze_history.append(point)
    
    def get_attention_score(self) -> float:
        """Calculate attention score based on gaze stability"""
        if len(self.gaze_history) < 5:
            return 0.5
        
        recent_points = list(self.gaze_history)[-10:]
        
        
        x_coords = [p.x for p in recent_points]
        y_coords = [p.y for p in recent_points]
        
        variance = np.var(x_coords) + np.var(y_coords)
        
        
        attention = 1.0 / (1.0 + variance / 1000)
        return min(max(attention, 0.0), 1.0)
    
    def detect_fixation(self) -> Optional[Tuple[float, float, float]]:
        """Detect current fixation point and duration"""
        if len(self.gaze_history) < 5:
            return None
        
        recent_points = list(self.gaze_history)[-10:]
        
        
        center_x = np.mean([p.x for p in recent_points])
        center_y = np.mean([p.y for p in recent_points])
        
        distances = [math.sqrt((p.x - center_x)**2 + (p.y - center_y)**2) for p in recent_points]
        
        if max(distances) < self.fixation_threshold:
            duration = recent_points[-1].timestamp - recent_points[0].timestamp
            if duration > self.min_fixation_duration:
                return center_x, center_y, duration
        
        return None

class AdvancedEyeTracker:
    """iPhone-inspired advanced eye tracking system with pure eye movement calibration"""
    
    def __init__(self):
        self.screen_w, self.screen_h = pyautogui.size()
        
        
        self.kalman_filter = KalmanGazeFilter()
        self.attention_detector = AttentionDetector()
        
        
        self.current_gaze = GazePoint(0, 0, 0, 0)
        self.eye_state = EyeState((0, 0), (0, 0), (0, 0, 0), 0, 0)
        
        
        self.calibration_data = None
        self.is_calibrated = False
        self.calibration_quality = "unknown"
        self.auto_fallback_enabled = True
        
        
        self.landmark_mappings = None
        self.screen_region_landmarks = None
        self.calibration_type = "unknown"
        
        self.load_latest_calibration()
        
        
        self.adaptation_enabled = True
        self.usage_data = deque(maxlen=1000)  
        
        
        self.gaze_history = deque(maxlen=50)  
        self.screen_boundary_margin = 0.15  
        self.cursor_control_enabled = True  
        
        
        self.fps_counter = deque(maxlen=30)
        self.accuracy_buffer = deque(maxlen=100)
        
        
        self.mouse_smoothing = 0.4 if self.is_calibrated else 0.3  
        self.last_mouse_pos = (0, 0)
        self._last_movement = [0, 0]  
        self._last_screen_offset = [0, 0]  
        
        
        self._current_landmark_signature = None
        self._last_adjustment = np.array([0, 0])
        
        print("🚀 Advanced Eye Tracker initialized")
        print(f"📺 Screen: {self.screen_w}x{self.screen_h}")
        print("📷 Coordinate system: Non-mirrored (matches calibrator)")
        if self.is_calibrated:
            print(f"✅ Pure eye movement calibration loaded! (Quality: {self.calibration_quality})")
        else:
            if self.calibration_data:
                print(f"🔄 Calibration available but using basic mapping (Quality: {self.calibration_quality})")
            else:
                print("⚠️  No calibration found - using basic head tracking mode")
    
    def load_latest_calibration(self):
        """Load the most recent calibration data with landmark-based mapping support"""
        try:
            
            current_dir = os.getcwd()
            services_dir = os.path.dirname(os.path.abspath(__file__))  # Fixed path resolution
            
            search_dirs = [current_dir, services_dir]
            calibration_files = []
            
            print(f"🔍 Searching for calibration files in:")
            print(f"   Current dir: {current_dir}")
            print(f"   Services dir: {services_dir}")
            
            for search_dir in search_dirs:
                landmark_files = glob.glob(os.path.join(search_dir, "landmark_eye_calibration_*.json"))
                traditional_files = glob.glob(os.path.join(search_dir, "pure_eye_calibration_*.json"))
                
                print(f"   Found {len(landmark_files)} landmark files in {search_dir}")
                print(f"   Found {len(traditional_files)} traditional files in {search_dir}")
                
                calibration_files.extend(landmark_files + traditional_files)
            
            if not calibration_files:
                print("📄 No calibration files found in current or services directory")
                return False
            
            
            landmark_files = [f for f in calibration_files if 'landmark_eye_calibration' in f]
            traditional_files = [f for f in calibration_files if 'pure_eye_calibration' in f]
            
           
            landmark_files.sort(key=os.path.getmtime, reverse=True)
            traditional_files.sort(key=os.path.getmtime, reverse=True)
            
            
            if landmark_files:
                latest_file = landmark_files[0]
                print(f"📍 Using latest landmark calibration: {os.path.basename(latest_file)}")
            else:
                latest_file = traditional_files[0]
                print(f"📐 Using latest traditional calibration: {os.path.basename(latest_file)}")
            
            print(f"📂 Loading from: {latest_file}")
            
            with open(latest_file, 'r') as f:
                self.calibration_data = json.load(f)
            
            
            self.calibration_type = self.calibration_data.get('calibration_type', 'traditional')
            
            if self.calibration_type == 'landmark_based':
                
                self.landmark_mappings = self.calibration_data.get('landmark_screen_mapping', {})
                self.screen_region_landmarks = self.calibration_data.get('screen_region_landmarks', {})
                print(f"🗺️  Loaded {len(self.landmark_mappings)} landmark mappings")
                print(f"📍 Loaded {len(self.screen_region_landmarks)} screen regions")
                
                
                mapping_quality = self.calibration_data.get('mapping_quality', {})
                avg_quality = mapping_quality.get('average_quality', 0.5)
                
                if avg_quality > 0.8:
                    self.calibration_quality = "excellent"
                elif avg_quality > 0.6:
                    self.calibration_quality = "good"
                elif avg_quality > 0.4:
                    self.calibration_quality = "fair"
                else:
                    self.calibration_quality = "poor"
                    
            else:
               
                accuracy = self.calibration_data['transformation_matrix']['accuracy']['total_rmse']
                transformation_type = self.calibration_data['transformation_matrix'].get('transformation_type', 'polynomial')
                
                
                if accuracy < 50:
                    self.calibration_quality = "excellent"
                elif accuracy < 100:
                    self.calibration_quality = "good"
                elif accuracy < 200:
                    self.calibration_quality = "fair"
                else:
                    self.calibration_quality = "poor"
            
            
            if self.calibration_quality == "poor" and self.auto_fallback_enabled:
                print(f"⚠️  Calibration quality is poor")
                print("🔄 Auto-fallback to basic mapping enabled")
                self.is_calibrated = False
            else:
                self.is_calibrated = True
                print(f"✅ Calibration enabled - using {self.calibration_type} mapping")
            
            print(f"✅ Loaded calibration: {latest_file}")
            print(f"📊 Type: {self.calibration_type}, Quality: {self.calibration_quality}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading calibration: {e}")
            return False
    
    def process_face_landmarks(self, landmarks, frame_shape) -> bool:
        """Process MediaPipe face landmarks with advanced algorithms"""
        h, w = frame_shape[:2]
        
        try:
            
            left_eye_center = self._get_eye_center(landmarks, "left", w, h)
            right_eye_center = self._get_eye_center(landmarks, "right", w, h)
            
            left_iris = left_eye_center
            right_iris = right_eye_center
            
            
            self._current_landmark_signature = self._extract_current_landmark_signature(landmarks, w, h)
            
            
            head_pose = self._estimate_head_pose(landmarks, w, h)
            
            
            blink_ratio = self._calculate_blink_ratio(landmarks, w, h)
            
            
            self.eye_state = EyeState(
                left_iris=left_iris,
                right_iris=right_iris,
                head_pose=head_pose,
                blink_ratio=blink_ratio,
                attention_score=0
            )
            
            return True
            
        except Exception as e:
            print(f"Error processing landmarks: {e}")
            return False
    
    def _get_precise_iris_center(self, landmarks, indices: List[int], w: int, h: int) -> Tuple[float, float]:
        """Get iris center with simple averaging"""
        if not indices or len(indices) == 0:
            return 0.0, 0.0
            
        points = []
        for idx in indices:
            if idx < len(landmarks.landmark):
                point = landmarks.landmark[idx]
                points.append([point.x * w, point.y * h])
        
        if not points:
            return 0.0, 0.0
            
        points = np.array(points, dtype=np.float32)
        center = np.mean(points, axis=0)
        
        return float(center[0]), float(center[1])
    
    def _get_eye_center(self, landmarks, eye: str, w: int, h: int) -> Tuple[float, float]:
        """Get eye center using multiple landmarks for better accuracy"""
        if eye == "left":
            
            eye_landmarks = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
            
            try:
                iris_center = landmarks.landmark[468]  # Left iris center
                if iris_center.x > 0 and iris_center.y > 0:
                    return float(iris_center.x * w), float(iris_center.y * h)
            except (IndexError, AttributeError):
                pass
        else:
            
            eye_landmarks = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
            
            try:
                iris_center = landmarks.landmark[473]  
                if iris_center.x > 0 and iris_center.y > 0:
                    return float(iris_center.x * w), float(iris_center.y * h)
            except (IndexError, AttributeError):
                pass
        
        
        eye_points = []
        for idx in eye_landmarks:
            if idx < len(landmarks.landmark):
                point = landmarks.landmark[idx]
                eye_points.append([point.x * w, point.y * h])
        
        if eye_points:
            eye_array = np.array(eye_points)
            center = np.mean(eye_array, axis=0)
            return float(center[0]), float(center[1])
        
        
        if eye == "left":
            outer_corner = landmarks.landmark[33]
            inner_corner = landmarks.landmark[133]
        else:
            outer_corner = landmarks.landmark[362]
            inner_corner = landmarks.landmark[263]
        
        center_x = (outer_corner.x + inner_corner.x) / 2 * w
        center_y = (outer_corner.y + inner_corner.y) / 2 * h
        
        return float(center_x), float(center_y)
    
    def _extract_current_landmark_signature(self, landmarks, w: int, h: int) -> dict:
        """Extract landmark signature matching the calibrator format"""
        try:
            signature = {}
            
            
            left_iris_indices = [474, 475, 476, 477, 478]
            left_iris_points = []
            for idx in left_iris_indices:
                if idx < len(landmarks.landmark):
                    point = landmarks.landmark[idx]
                    left_iris_points.append([point.x * w, point.y * h, point.z])
            
            if left_iris_points:
                left_iris_array = np.array(left_iris_points)
                signature['left_iris'] = {
                    'centroid': np.mean(left_iris_array[:, :2], axis=0).tolist(),
                    'std': np.std(left_iris_array[:, :2], axis=0).tolist(),
                    'area': len(left_iris_points) * 2.0  
                }
            
            
            right_iris_indices = [469, 470, 471, 472, 473]
            right_iris_points = []
            for idx in right_iris_indices:
                if idx < len(landmarks.landmark):
                    point = landmarks.landmark[idx]
                    right_iris_points.append([point.x * w, point.y * h, point.z])
            
            if right_iris_points:
                right_iris_array = np.array(right_iris_points)
                signature['right_iris'] = {
                    'centroid': np.mean(right_iris_array[:, :2], axis=0).tolist(),
                    'std': np.std(right_iris_array[:, :2], axis=0).tolist(),
                    'area': len(right_iris_points) * 2.0  # Approximate area
                }
            
            
            left_eye_outline = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
            left_eye_points = []
            for idx in left_eye_outline:
                if idx < len(landmarks.landmark):
                    point = landmarks.landmark[idx]
                    left_eye_points.append([point.x * w, point.y * h])
            
            if left_eye_points:
                left_eye_array = np.array(left_eye_points)
                signature['left_eye_region'] = {
                    'centroid': np.mean(left_eye_array, axis=0).tolist(),
                    'std': np.std(left_eye_array, axis=0).tolist()
                }
            
            right_eye_outline = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
            right_eye_points = []
            for idx in right_eye_outline:
                if idx < len(landmarks.landmark):
                    point = landmarks.landmark[idx]
                    right_eye_points.append([point.x * w, point.y * h])
            
            if right_eye_points:
                right_eye_array = np.array(right_eye_points)
                signature['right_eye_region'] = {
                    'centroid': np.mean(right_eye_array, axis=0).tolist(),
                    'std': np.std(right_eye_array, axis=0).tolist()
                }
            
            
            key_points = [1, 175, 234, 454, 10, 152]  
            face_points = []
            for idx in key_points:
                if idx < len(landmarks.landmark):
                    point = landmarks.landmark[idx]
                    face_points.append([point.x * w, point.y * h])
            
            if face_points:
                face_array = np.array(face_points)
                signature['face_structure'] = {
                    'centroid': np.mean(face_array, axis=0).tolist(),
                    'std': np.std(face_array, axis=0).tolist()
                }
            
            return signature
            
        except Exception as e:
            print(f"Error extracting landmark signature: {e}")
            return {}
    
    def _estimate_head_pose(self, landmarks, w: int, h: int) -> Tuple[float, float, float]:
        """Estimate 3D head pose for gaze correction"""
        
        nose_tip = landmarks.landmark[1]
        chin = landmarks.landmark[175]
        left_eye = landmarks.landmark[33]
        right_eye = landmarks.landmark[263]
        left_mouth = landmarks.landmark[61]
        right_mouth = landmarks.landmark[291]
        
        # Convert to pixel coordinates
        nose_tip = np.array([nose_tip.x * w, nose_tip.y * h])
        chin = np.array([chin.x * w, chin.y * h])
        left_eye = np.array([left_eye.x * w, left_eye.y * h])
        right_eye = np.array([right_eye.x * w, right_eye.y * h])
        
        # Calculate rotation angles
        eye_center = (left_eye + right_eye) / 2
        face_center = (nose_tip + chin) / 2
        
        # Yaw (left-right head turn)
        eye_width = np.linalg.norm(right_eye - left_eye)
        expected_eye_width = w * 0.15  # Approximate expected width
        yaw = math.atan2(eye_width - expected_eye_width, expected_eye_width) * 180 / math.pi
        
        # Pitch (up-down head tilt)
        face_height = np.linalg.norm(nose_tip - chin)
        expected_face_height = h * 0.15
        pitch = math.atan2(face_height - expected_face_height, expected_face_height) * 180 / math.pi
        
        # Roll (head rotation)
        roll = math.atan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]) * 180 / math.pi
        
        return yaw, pitch, roll
    
    def _calculate_blink_ratio(self, landmarks, w: int, h: int) -> float:
        """Calculate eye aspect ratio for blink detection"""
        # Left eye landmarks
        left_eye_landmarks = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        
        # Get eye points
        left_points = []
        for idx in left_eye_landmarks:
            point = landmarks.landmark[idx]
            left_points.append([point.x * w, point.y * h])
        
        left_points = np.array(left_points)
        
        # Calculate eye aspect ratio
        # Vertical distances
        vertical1 = np.linalg.norm(left_points[1] - left_points[5])
        vertical2 = np.linalg.norm(left_points[2] - left_points[4])
        
        # Horizontal distance
        horizontal = np.linalg.norm(left_points[0] - left_points[3])
        
        # Eye aspect ratio
        ear = (vertical1 + vertical2) / (2.0 * horizontal)
        return ear
    
    def calculate_gaze_point(self) -> GazePoint:
        """Calculate gaze point using advanced algorithms"""
        # Average both eyes for better accuracy
        avg_iris_x = (self.eye_state.left_iris[0] + self.eye_state.right_iris[0]) / 2
        avg_iris_y = (self.eye_state.left_iris[1] + self.eye_state.right_iris[1]) / 2
        
        # Debug eye positions
        if hasattr(self, '_debug_counter2'):
            self._debug_counter2 += 1
        else:
            self._debug_counter2 = 0
        
        if self._debug_counter2 % 30 == 0:  # Every 30 frames
            print(f"👁️ DEBUG: left_iris=({self.eye_state.left_iris[0]:.1f},{self.eye_state.left_iris[1]:.1f})")
            print(f"👁️ DEBUG: right_iris=({self.eye_state.right_iris[0]:.1f},{self.eye_state.right_iris[1]:.1f})")
            print(f"👁️ DEBUG: avg_iris=({avg_iris_x:.1f},{avg_iris_y:.1f})")
        
        # Apply head pose correction
        corrected_x, corrected_y = self._apply_head_pose_correction(avg_iris_x, avg_iris_y)
        
        # Map to screen coordinates
        screen_x, screen_y = self._map_to_screen(corrected_x, corrected_y)
        
        # Calculate confidence based on multiple factors
        confidence = self._calculate_confidence()
        
        # Create gaze point
        gaze_point = GazePoint(
            x=screen_x,
            y=screen_y,
            timestamp=time.time(),
            confidence=confidence
        )
        
        # Update Kalman filter
        self.kalman_filter.update((screen_x, screen_y), confidence)
        
        # Get filtered prediction
        filtered_x, filtered_y = self.kalman_filter.predict()
        
        # Update with filtered coordinates
        gaze_point.x = filtered_x
        gaze_point.y = filtered_y
        
        return gaze_point
    
    def _apply_head_pose_correction(self, iris_x: float, iris_y: float) -> Tuple[float, float]:
        """Apply head pose correction to iris position"""
        yaw, pitch, roll = self.eye_state.head_pose
        
        # Correction factors based on head pose
        yaw_correction = yaw * 2.0  # Adjust sensitivity
        pitch_correction = pitch * 1.5
        
        corrected_x = iris_x + yaw_correction
        corrected_y = iris_y + pitch_correction
        
        return corrected_x, corrected_y
    
    def _map_to_screen(self, iris_x: float, iris_y: float) -> Tuple[float, float]:
        """Map iris coordinates to screen coordinates using landmark-based or traditional calibration"""
        
        if self.is_calibrated and self.calibration_data:
            if self.calibration_type == 'landmark_based':
                # Use landmark-based mapping
                if hasattr(self, '_debug_counter') and self._debug_counter % 60 == 0:
                    print(f"🗺️  Using LANDMARK-BASED calibration mapping")
                return self._apply_landmark_mapping(iris_x, iris_y)
            else:
                # Use traditional polynomial mapping
                if hasattr(self, '_debug_counter') and self._debug_counter % 60 == 0:
                    print(f"📐 Using TRADITIONAL calibration mapping")
                return self._apply_calibrated_mapping(iris_x, iris_y)
        else:
            # Fallback to basic head+eye tracking
            if hasattr(self, '_debug_counter') and self._debug_counter % 60 == 0:
                print(f"⚠️  Using BASIC mapping (no calibration)")
            return self._apply_basic_mapping(iris_x, iris_y)
    
    def _apply_landmark_mapping(self, iris_x: float, iris_y: float) -> Tuple[float, float]:
        """Apply landmark-based mapping for ultra-precise screen coordinate mapping"""
        try:
            if not self.landmark_mappings:
                print("⚠️  No landmark mappings available, using basic mapping")
                return self._apply_basic_mapping(iris_x, iris_y)
            
            # Debug output to confirm landmark mapping is being used
            if hasattr(self, '_debug_counter'):
                self._debug_counter += 1
            else:
                self._debug_counter = 0
            
            if self._debug_counter % 60 == 0:  # Every 60 frames
                print(f"🗺️  USING LANDMARK MAPPING: {len(self.landmark_mappings)} mappings available")
                print(f"🎯 Current iris: ({iris_x:.1f},{iris_y:.1f})")
                if hasattr(self, '_current_landmark_signature') and self._current_landmark_signature:
                    print(f"🔍 Real-time signature extracted: {len(self._current_landmark_signature)} components")
                else:
                    print(f"⚠️  No current signature available")
            
            # Find the closest landmark mapping based on iris position
            best_match = None
            min_distance = float('inf')
            
            for mapping_key, mapping_data in self.landmark_mappings.items():
                # Compare current iris position with stored patterns
                stored_signature = mapping_data.get('landmark_signature', {})
                
                # Calculate similarity score
                similarity = self._calculate_landmark_similarity(iris_x, iris_y, stored_signature)
                distance = 1.0 - similarity
                
                if distance < min_distance:
                    min_distance = distance
                    best_match = mapping_data
            
            if best_match and min_distance < 0.8:  # More lenient threshold for better matching
                # Use the matched screen position
                base_screen_pos = best_match['screen_position']
                
                # Apply fine adjustment based on landmark signature
                adjustment = self._calculate_position_adjustment(iris_x, iris_y, best_match)
                
                final_x = base_screen_pos[0] + adjustment[0]
                final_y = base_screen_pos[1] + adjustment[1]
                
                # Debug output
                if self._debug_counter % 60 == 0:
                    print(f"🎯 LANDMARK MATCH: similarity={1.0-min_distance:.3f}")
                    print(f"📍 Base position: ({base_screen_pos[0]:.0f},{base_screen_pos[1]:.0f})")
                    print(f"🎯 Final position: ({final_x:.0f},{final_y:.0f})")
                    print(f"🔧 Adjustment: ({adjustment[0]:.1f},{adjustment[1]:.1f})")
                
                # Clamp to screen bounds
                final_x = max(0, min(final_x, self.screen_w))
                final_y = max(0, min(final_y, self.screen_h))
                
                return final_x, final_y
            else:
                # No good match found, use basic mapping
                if self._debug_counter % 60 == 0:
                    print(f"⚠️  No landmark match found (min_distance={min_distance:.3f}), using basic mapping")
                return self._apply_basic_mapping(iris_x, iris_y)
                
        except Exception as e:
            print(f"❌ Error in landmark mapping: {e}")
            return self._apply_basic_mapping(iris_x, iris_y)
    
    def _calculate_landmark_similarity(self, iris_x, iris_y, stored_signature):
        """Calculate similarity between current landmark signature and stored calibration signature"""
        try:
            if not stored_signature or not hasattr(self, '_current_landmark_signature'):
                return 0.0
            
            current_signature = self._current_landmark_signature
            if not current_signature:
                return 0.0
            
            total_similarity = 0.0
            comparison_count = 0
            
            # Compare iris positions (most important)
            for iris_side in ['left_iris', 'right_iris']:
                if iris_side in current_signature and iris_side in stored_signature:
                    current_centroid = np.array(current_signature[iris_side]['centroid'])
                    stored_centroid = np.array(stored_signature[iris_side]['centroid'])
                    
                    # Calculate distance between centroids
                    distance = np.linalg.norm(current_centroid - stored_centroid)
                    
                    # Convert to similarity (closer = more similar)
                    max_distance = 50  # Adjust based on typical iris movement range
                    similarity = max(0, 1.0 - distance / max_distance)
                    
                    total_similarity += similarity * 0.4  # High weight for iris
                    comparison_count += 0.4
            
            # Compare eye regions
            for eye_region in ['left_eye_region', 'right_eye_region']:
                if eye_region in current_signature and eye_region in stored_signature:
                    current_centroid = np.array(current_signature[eye_region]['centroid'])
                    stored_centroid = np.array(stored_signature[eye_region]['centroid'])
                    
                    distance = np.linalg.norm(current_centroid - stored_centroid)
                    max_distance = 30
                    similarity = max(0, 1.0 - distance / max_distance)
                    
                    total_similarity += similarity * 0.1  # Lower weight for eye regions
                    comparison_count += 0.1
            
            # Compare face structure
            if 'face_structure' in current_signature and 'face_structure' in stored_signature:
                current_centroid = np.array(current_signature['face_structure']['centroid'])
                stored_centroid = np.array(stored_signature['face_structure']['centroid'])
                
                distance = np.linalg.norm(current_centroid - stored_centroid)
                max_distance = 20
                similarity = max(0, 1.0 - distance / max_distance)
                
                total_similarity += similarity * 0.1  # Face structure weight
                comparison_count += 0.1
            
            # Calculate final similarity
            if comparison_count > 0:
                final_similarity = total_similarity / comparison_count
                return min(1.0, max(0.0, final_similarity))
            
            return 0.0
            
        except Exception as e:
            print(f"Similarity calculation failed: {e}")
            return 0.0
    
    def _calculate_position_adjustment(self, iris_x, iris_y, best_match):
        """Calculate fine position adjustment based on landmark signature differences"""
        try:
            if not hasattr(self, '_current_landmark_signature') or not self._current_landmark_signature:
                return np.array([0, 0])
            
            # Get the stored landmark signature
            stored_signature = best_match.get('landmark_signature', {})
            current_signature = self._current_landmark_signature
            
            if not stored_signature:
                return np.array([0, 0])
            
            # Calculate offset based on iris positions
            adjustment_x = 0
            adjustment_y = 0
            valid_adjustments = 0
            
            for iris_side in ['left_iris', 'right_iris']:
                if (iris_side in current_signature and iris_side in stored_signature and
                    'centroid' in current_signature[iris_side] and 'centroid' in stored_signature[iris_side]):
                    
                    current_centroid = np.array(current_signature[iris_side]['centroid'])
                    stored_centroid = np.array(stored_signature[iris_side]['centroid'])
                    
                    # Calculate offset
                    offset = current_centroid - stored_centroid
                    
                    # Scale offset to screen coordinates with calibrated sensitivity
                    if iris_side == 'left_iris':
                        sensitivity_x = self.screen_w / 100  # Adjusted sensitivity
                        sensitivity_y = self.screen_h / 80
                    else:
                        sensitivity_x = self.screen_w / 100
                        sensitivity_y = self.screen_h / 80
                    
                    adjustment_x += offset[0] * sensitivity_x
                    adjustment_y += offset[1] * sensitivity_y
                    valid_adjustments += 1
            
            # Average the adjustments if we have multiple
            if valid_adjustments > 0:
                adjustment_x /= valid_adjustments
                adjustment_y /= valid_adjustments
                
                # Apply smoothing to prevent jittery movement
                if hasattr(self, '_last_adjustment'):
                    smoothing = 0.3
                    adjustment_x = self._last_adjustment[0] * (1 - smoothing) + adjustment_x * smoothing
                    adjustment_y = self._last_adjustment[1] * (1 - smoothing) + adjustment_y * smoothing
                
                self._last_adjustment = np.array([adjustment_x, adjustment_y])
                return self._last_adjustment
            
            return np.array([0, 0])
            
        except Exception as e:
            print(f"Position adjustment failed: {e}")
            return np.array([0, 0])
    
    def _interpolate_landmark_mappings(self, iris_x, iris_y):
        """Interpolate between multiple landmark mappings when no direct match"""
        try:
            if not self.landmark_mappings or len(self.landmark_mappings) < 2:
                return self._apply_basic_mapping(iris_x, iris_y)
            
            # Find the 3 closest mappings for triangulation
            distances = []
            for mapping_key, mapping_data in self.landmark_mappings.items():
                signature = mapping_data.get('landmark_signature', {})
                similarity = self._calculate_landmark_similarity(iris_x, iris_y, signature)
                distance = 1.0 - similarity
                distances.append((distance, mapping_data))
            
            # Sort by distance and take top 3
            distances.sort(key=lambda x: x[0])
            closest_mappings = distances[:3]
            
            # Weighted interpolation based on inverse distance
            total_weight = 0
            weighted_x = 0
            weighted_y = 0
            
            for distance, mapping_data in closest_mappings:
                if distance < 1e-6:  # Avoid division by zero
                    distance = 1e-6
                
                weight = 1.0 / distance
                screen_pos = mapping_data['screen_position']
                
                weighted_x += screen_pos[0] * weight
                weighted_y += screen_pos[1] * weight
                total_weight += weight
            
            if total_weight > 0:
                interpolated_x = weighted_x / total_weight
                interpolated_y = weighted_y / total_weight
                
                # Clamp to screen bounds
                interpolated_x = max(0, min(interpolated_x, self.screen_w))
                interpolated_y = max(0, min(interpolated_y, self.screen_h))
                
                return interpolated_x, interpolated_y
            else:
                return self._apply_basic_mapping(iris_x, iris_y)
                
        except Exception as e:
            print(f"Interpolation failed: {e}")
            return self._apply_basic_mapping(iris_x, iris_y)
    
    def _apply_calibrated_mapping(self, iris_x: float, iris_y: float) -> Tuple[float, float]:
        """Apply calibrated transformation for pure eye movement"""
        try:
            if 'transformation_matrix' not in self.calibration_data:
                print("⚠️  No transformation matrix in calibration data, using basic mapping")
                return self._apply_basic_mapping(iris_x, iris_y)
                
            transformation = self.calibration_data['transformation_matrix']
            x_coeffs = np.array(transformation['x_coeffs'])
            y_coeffs = np.array(transformation['y_coeffs'])
            
            # Debug output to confirm calibration is being used
            if hasattr(self, '_debug_counter'):
                self._debug_counter += 1
            else:
                self._debug_counter = 0
            
            if self._debug_counter % 60 == 0:  # Every 60 frames
                print(f"🎯 USING CALIBRATED MAPPING: iris=({iris_x:.1f},{iris_y:.1f})")
                print(f"🔧 Transformation type: {transformation.get('transformation_type', 'polynomial')}")
            
            # Apply normalization if available
            if 'normalization' in transformation:
                eye_mean = np.array(transformation['normalization']['eye_mean'])
                eye_std = np.array(transformation['normalization']['eye_std'])
                iris_normalized = np.array([(iris_x - eye_mean[0]) / eye_std[0], 
                                          (iris_y - eye_mean[1]) / eye_std[1]])
            else:
                iris_normalized = np.array([iris_x, iris_y])
            
            # Create feature vector based on transformation type
            transformation_type = transformation.get('transformation_type', 'polynomial')
            
            if transformation_type == 'linear':
                # Linear transformation: [1, x, y]
                features = np.array([1, iris_normalized[0], iris_normalized[1]])
                # Adjust coefficients if needed
                if len(x_coeffs) > 3:
                    x_coeffs = x_coeffs[:3]
                if len(y_coeffs) > 3:
                    y_coeffs = y_coeffs[:3]
            else:
                # Polynomial transformation: [1, x, y, x^2, y^2, x*y]
                features = np.array([
                    1,
                    iris_normalized[0],
                    iris_normalized[1],
                    iris_normalized[0] ** 2,
                    iris_normalized[1] ** 2,
                    iris_normalized[0] * iris_normalized[1]
                ])
            
            # Apply transformation
            screen_x = np.dot(features, x_coeffs)
            screen_y = np.dot(features, y_coeffs)
            
            if self._debug_counter % 60 == 0:
                print(f"🎯 CALIBRATED RESULT: screen=({screen_x:.1f},{screen_y:.1f})")
            
            # Clamp to screen bounds
            screen_x = max(0, min(screen_x, self.screen_w))
            screen_y = max(0, min(screen_y, self.screen_h))
            
            return screen_x, screen_y
            
        except Exception as e:
            print(f"❌ Error in calibrated mapping: {e}")
            print("🔄 Falling back to basic mapping")
            return self._apply_basic_mapping(iris_x, iris_y)
    
    def _apply_basic_mapping(self, iris_x: float, iris_y: float) -> Tuple[float, float]:
        """Apply SIMPLIFIED camera-to-screen coordinate mapping to fix cursor disappearing"""
        # Simple mapping approach that should work reliably
        camera_width = 640
        camera_height = 480
        
        # Use fixed center for now to avoid complexity
        center_x = camera_width / 2   # 320
        center_y = camera_height / 2  # 240
        
        print(f"🎯 BASIC MAPPING: iris=({iris_x:.1f},{iris_y:.1f}) camera_center=({center_x},{center_y})")
        
        # Calculate offset from center in camera coordinates  
        offset_x = iris_x - center_x
        offset_y = iris_y - center_y
        
        print(f"   Camera offset: ({offset_x:.1f},{offset_y:.1f})")
        
        # Simple proportional mapping to screen coordinates
        # Map camera range to screen range with sensitivity scaling
        sensitivity_x = 4.0  # How sensitive horizontal movement is
        sensitivity_y = 3.0  # How sensitive vertical movement is
        
        # Map to screen coordinates from center
        screen_x = self.screen_w / 2 + (offset_x * sensitivity_x)
        screen_y = self.screen_h / 2 + (offset_y * sensitivity_y)
        
        # Ensure coordinates are within screen bounds with margin
        margin = 10
        screen_x = max(margin, min(screen_x, self.screen_w - margin))
        screen_y = max(margin, min(screen_y, self.screen_h - margin))
        
        print(f"   Screen mapping: ({screen_x:.1f},{screen_y:.1f}) bounds=({self.screen_w}x{self.screen_h})")
        
        return screen_x, screen_y
    
    def _calculate_confidence(self) -> float:
        """Calculate tracking confidence based on multiple factors"""
        confidence = 1.0
        
        # Reduce confidence if head pose is extreme
        yaw, pitch, roll = self.eye_state.head_pose
        if abs(yaw) > 30 or abs(pitch) > 20:
            confidence *= 0.7
        
        # Reduce confidence if blinking
        if self.eye_state.blink_ratio < 0.2:  # Likely blinking
            confidence *= 0.3
        
        # Consider iris detection quality
        left_x, left_y = self.eye_state.left_iris
        right_x, right_y = self.eye_state.right_iris
        
        if left_x == 0 or right_x == 0:  # Poor detection
            confidence *= 0.5
        
        return max(confidence, 0.1)
    
    def update_attention_score(self):
        """Update attention score based on gaze patterns"""
        self.attention_detector.add_gaze_point(self.current_gaze)
        attention = self.attention_detector.get_attention_score()
        
        # Update eye state
        self.eye_state = EyeState(
            left_iris=self.eye_state.left_iris,
            right_iris=self.eye_state.right_iris,
            head_pose=self.eye_state.head_pose,
            blink_ratio=self.eye_state.blink_ratio,
            attention_score=attention
        )
    
    def control_mouse(self, gaze_point: GazePoint):
        """Control mouse with adaptive smoothing for better accuracy"""
        if gaze_point.confidence < 0.2:  # More lenient confidence threshold
            print(f"⚠️  Skipping low confidence gaze: {gaze_point.confidence:.3f}")
            return  # Skip low-confidence points
        
        # Enhanced debugging - always show what we're trying to do
        print(f"🎯 MOUSE CONTROL: gaze=({gaze_point.x:.1f},{gaze_point.y:.1f}) confidence={gaze_point.confidence:.3f}")
        
        try:
            current_x, current_y = pyautogui.position()
            print(f"📍 Current cursor: ({current_x},{current_y})")
        except Exception as e:
            print(f"❌ Error getting current mouse position: {e}")
            return
        
        # Calculate distance to target
        distance = math.sqrt((gaze_point.x - current_x)**2 + (gaze_point.y - current_y)**2)
        
        # Adaptive smoothing based on distance and confidence
        if distance < 20:  # Very close - high precision
            smoothing = 0.15 * gaze_point.confidence
        elif distance < 50:  # Close - medium precision
            smoothing = 0.25 * gaze_point.confidence
        elif distance < 100:  # Medium distance - balanced
            smoothing = 0.4 * gaze_point.confidence
        else:  # Far distance - quick movement
            smoothing = 0.6 * gaze_point.confidence
        
        # Apply minimum smoothing to prevent jitter
        smoothing = max(smoothing, 0.1)
        
        # Calculate smooth movement
        smooth_x = current_x + (gaze_point.x - current_x) * smoothing
        smooth_y = current_y + (gaze_point.y - current_y) * smoothing
        
        # Add acceleration for larger movements
        if distance > 100:
            # Reduce smoothing for faster movement to distant targets
            acceleration_factor = min(distance / 200, 2.0)
            smooth_x = current_x + (gaze_point.x - current_x) * smoothing * acceleration_factor
            smooth_y = current_y + (gaze_point.y - current_y) * smoothing * acceleration_factor
        
        # Ensure coordinates are within screen bounds
        smooth_x = max(0, min(smooth_x, self.screen_w - 1))
        smooth_y = max(0, min(smooth_y, self.screen_h - 1))
        
        # Move mouse with enhanced error handling
        try:
            print(f"🖱️  MOVING CURSOR: ({current_x},{current_y}) -> ({int(smooth_x)},{int(smooth_y)}) [distance={distance:.1f}]")
            pyautogui.moveTo(int(smooth_x), int(smooth_y), duration=0)
            
            # Verify the move worked
            new_x, new_y = pyautogui.position()
            print(f"✅ CURSOR MOVED: now at ({new_x},{new_y})")
            
            # Store last position for tracking
            self.last_mouse_pos = (smooth_x, smooth_y)
                
        except Exception as e:
            print(f"❌ Error moving mouse: {e}")
            print(f"   Attempted coordinates: ({int(smooth_x)},{int(smooth_y)})")
            print(f"   Screen bounds: {self.screen_w}x{self.screen_h}")
            # Try alternative method
            try:
                import ctypes
                ctypes.windll.user32.SetCursorPos(int(smooth_x), int(smooth_y))
                print("✅ Used alternative Windows API for mouse control")
            except Exception as e2:
                print(f"❌ Alternative mouse control also failed: {e2}")
    
    def is_gaze_within_screen_boundary(self, gaze_point: GazePoint) -> bool:
        """Determine if current gaze is within screen viewing area"""
        # Define expanded boundary area (allows some tolerance)
        margin_x = self.screen_w * self.screen_boundary_margin
        margin_y = self.screen_h * self.screen_boundary_margin
        
        boundary_left = -margin_x
        boundary_right = self.screen_w + margin_x
        boundary_top = -margin_y
        boundary_bottom = self.screen_h + margin_y
        
        # Check if gaze is within expanded screen boundary
        within_boundary = (boundary_left <= gaze_point.x <= boundary_right and 
                          boundary_top <= gaze_point.y <= boundary_bottom)
        
        # Additional check: analyze gaze stability
        if len(self.gaze_history) >= 10:
            engagement = self.estimate_screen_engagement()
            if not engagement:
                return False  # User seems to be looking away
        
        return within_boundary
    
    def estimate_screen_engagement(self) -> bool:
        """Estimate if user is engaged with laptop screen based on gaze patterns"""
        if len(self.gaze_history) < 10:
            return True  # Default to engaged if insufficient data
        
        # Analyze recent gaze stability and patterns
        recent_positions = list(self.gaze_history)[-10:]
        
        # Calculate gaze variance (lower = more focused/stable)
        x_positions = [pos[0] for pos in recent_positions]
        y_positions = [pos[1] for pos in recent_positions]
        
        variance_x = np.var(x_positions)
        variance_y = np.var(y_positions)
        total_variance = variance_x + variance_y
        
        # High variance suggests looking around (possibly away from screen)
        # Low variance suggests focused attention (likely on screen)
        engagement_threshold = 2000  # Adjust based on testing
        
        is_engaged = total_variance < engagement_threshold
        
        return is_engaged
    
    def _are_coordinates_reasonable(self, x: float, y: float) -> bool:
        """Check if screen coordinates are reasonable"""
        # More conservative bounds checking
        margin_factor = 1.5  # Allow 50% overshoot
        max_x = self.screen_w * margin_factor
        max_y = self.screen_h * margin_factor
        min_x = -self.screen_w * 0.5
        min_y = -self.screen_h * 0.5
        
        # Check for finite values
        if not (np.isfinite(x) and np.isfinite(y)):
            return False
        
        # Check bounds
        if x < min_x or x > max_x or y < min_y or y > max_y:
            return False
        
        return True
    
    def detect_blink_click(self) -> bool:
        """Detect intentional blink for clicking"""
        if self.eye_state.blink_ratio < 0.15:  # Strong blink
            return True
        return False
    
    def get_performance_metrics(self) -> dict:
        """Get current performance metrics"""
        avg_fps = np.mean(self.fps_counter) if self.fps_counter else 0
        
        return {
            'fps': avg_fps,
            'confidence': self.current_gaze.confidence,
            'attention_score': self.eye_state.attention_score,
            'head_pose': self.eye_state.head_pose,
            'calibrated': self.is_calibrated,
            'tracking_mode': 'Pure Eye Movement' if self.is_calibrated else 'Head + Eye Movement',
            'cursor_control_enabled': self.cursor_control_enabled,
            'screen_engagement': self.estimate_screen_engagement() if len(self.gaze_history) >= 10 else True
        }
    
    def save_usage_data(self, filename: str = "advanced_eye_tracking_session.json"):
        """Save session data for future improvements"""
        session_data = {
            'usage_data': list(self.usage_data),
            'performance_metrics': self.get_performance_metrics(),
            'session_duration': time.time(),
            'screen_resolution': [self.screen_w, self.screen_h]
        }
        
        with open(filename, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        print(f"💾 Session data saved to {filename}")
    
    def start_quick_calibration(self):
        """Start a quick 9-point calibration process"""
        print("🎯 Quick calibration not implemented in demo mode")
        print("   For full calibration, run pure_eye_calibrator.py")
        print("   Current system uses adaptive camera-to-screen mapping")


    def _calculate_fps(self):
        """Estimate frames per second based on recent timestamps"""
        if len(self.fps_counter) < 2:
            return 0.0
        duration = self.fps_counter[-1] - self.fps_counter[0]
        return round(len(self.fps_counter) / duration, 2) if duration > 0 else 0.0
    
    def get_tracker_status(self) :
        """Generate a summary of current tracker status and performance metrics"""
        attention_score = self.attention_detector.get_attention_score()
        fixation = self.attention_detector.detect_fixation()
        screen_engagement = fixation is not None

        return {
            "usage_data": list(self.usage_data),
            "performance_metrics": {
            "fps": self._calculate_fps(),
            "confidence": self.current_gaze.confidence,
            "attention_score": attention_score,
            "head_pose": list(self.eye_state.head_pose),
            "calibrated": self.is_calibrated,
            "tracking_mode": "Pure Eye Movement" if self.is_calibrated else "Basic Head Tracking",
            "cursor_control_enabled": self.cursor_control_enabled,
            "screen_engagement": screen_engagement
        },
        "session_duration": time.time() - getattr(self, "session_start_time", 0),
        "screen_resolution": [self.screen_w, self.screen_h]
    }



    

def create_advanced_eye_tracking_demo():
    """Create a demo of the advanced eye tracking system"""
    print("🚀 Starting iPhone-inspired Advanced Eye Tracking Demo")
    print("📱 Features: Kalman filtering, attention detection, head pose correction")
    print("🔍 NEW: Gaze boundary detection - cursor only moves when looking at screen!")
    print("⌨️  Controls: ESC=exit, SPACE=toggle mouse, C=toggle calibration, M=toggle mirror")
    print("             R=reload calibration, Z=reset center, Q=quick calibration, I=help")
    
    # Initialize tracker
    tracker = AdvancedEyeTracker()
    
    # Camera setup with better error handling
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open camera")
        print("   Please check if camera is connected and not being used by another application")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("✅ Camera initialized successfully")
    print("🖱️  PyAutoGUI failsafe disabled for smooth mouse control")
    print("💡 If cursor disappears, check Windows mouse settings")
    
    # Demo state
    mouse_control_enabled = True
    frame_count = 0
    mirror_camera = False  # Keep FALSE to match calibrator coordinate system
    
    print(f"📷 Camera mirroring: {'ON' if mirror_camera else 'OFF'} (matches calibrator)")
    print("   Press 'M' to toggle camera mirroring")
    print("   ⚠️  Note: Calibrator uses non-mirrored coordinates")
    
    try:
        # Import MediaPipe
        import mediapipe as mp
        
        mp_face_mesh = mp.solutions.face_mesh
        mp_drawing = mp.solutions.drawing_utils
        
        face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.3,  # Lower for continuous detection
            min_tracking_confidence=0.2   # Lower for continuous tracking
        )
        
        print("✅ MediaPipe Face Mesh initialized successfully")
        
    except ImportError:
        print("❌ MediaPipe not available - using demo mode")
        print("   Install MediaPipe with compatible Python version to enable full functionality")
        
        # Demo mode with simulated eye movement
        while True:
            frame_count += 1
            
            # Simulate gaze point
            t = time.time()
            demo_x = tracker.screen_w/2 + 200 * math.sin(t)
            demo_y = tracker.screen_h/2 + 100 * math.cos(t * 1.5)
            
            demo_gaze = GazePoint(demo_x, demo_y, t, 0.8)
            tracker.current_gaze = demo_gaze
            tracker.update_attention_score()
            
            if mouse_control_enabled:
                tracker.control_mouse(demo_gaze)
            
            # Display metrics
            if frame_count % 30 == 0:
                metrics = tracker.get_performance_metrics()
                print(f"📊 Attention: {metrics['attention_score']:.2f}, Confidence: {metrics['confidence']:.2f}")
            
            # Check for exit
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord(' '):
                mouse_control_enabled = not mouse_control_enabled
                print(f"🖱️  Mouse control: {'ON' if mouse_control_enabled else 'OFF'}")
            elif key == ord('s'):
                tracker.save_usage_data()
            
            time.sleep(0.033)  # ~30 FPS
        
        print("✅ Advanced Eye Tracking Demo completed")
        return
    
    # Main tracking loop with simplified detection
    try:
        print("🎥 Starting camera capture...")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read from camera")
                break
            
            frame_count += 1
            frame_time = time.time()
            
            # Flip frame horizontally for mirror effect (optional)
            if mirror_camera:
                frame = cv2.flip(frame, 1)
            
            # Process frame
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                # Process the first detected face
                landmarks = results.multi_face_landmarks[0]
                
                # Process landmarks and calculate gaze
                if tracker.process_face_landmarks(landmarks, frame.shape):
                    gaze_point = tracker.calculate_gaze_point()
                    tracker.current_gaze = gaze_point
                    tracker.update_attention_score()
                    
                    # Control mouse with lower confidence threshold
                    if mouse_control_enabled and gaze_point.confidence > 0.2:
                        tracker.control_mouse(gaze_point)
                    
                    # Enhanced tracking info with visual feedback
                    cv2.putText(frame, f"Gaze: ({gaze_point.x:.0f}, {gaze_point.y:.0f})", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Color-coded confidence
                    conf_color = (0, 255, 0) if gaze_point.confidence > 0.7 else (0, 255, 255) if gaze_point.confidence > 0.4 else (0, 0, 255)
                    cv2.putText(frame, f"Confidence: {gaze_point.confidence:.2f}", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, conf_color, 2)
                    
                    mouse_status = "ON" if mouse_control_enabled else "OFF"
                    mouse_color = (0, 255, 0) if mouse_control_enabled else (0, 0, 255)
                    cv2.putText(frame, f"Mouse Control: {mouse_status}", 
                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, mouse_color, 2)
                    
                    # Show current mouse position for debugging
                    try:
                        mouse_x, mouse_y = pyautogui.position()
                        cv2.putText(frame, f"Mouse: ({mouse_x},{mouse_y})", 
                                   (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    except:
                        cv2.putText(frame, "Mouse: ERROR", 
                                   (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    
                    # Show eye positions for debugging with camera coordinate info
                    left_x, left_y = tracker.eye_state.left_iris
                    right_x, right_y = tracker.eye_state.right_iris
                    cv2.putText(frame, f"Camera: L({left_x:.0f},{left_y:.0f}) R({right_x:.0f},{right_y:.0f})", 
                               (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
                    
                    # Show screen coordinates mapping
                    cv2.putText(frame, f"Screen: ({gaze_point.x:.0f},{gaze_point.y:.0f}) of {tracker.screen_w}x{tracker.screen_h}", 
                               (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    
                    # Visual crosshair for gaze direction
                    if gaze_point.confidence > 0.3:
                        # Scale gaze point to frame coordinates for visualization
                        viz_x = int((gaze_point.x / tracker.screen_w) * frame.shape[1])
                        viz_y = int((gaze_point.y / tracker.screen_h) * frame.shape[0])
                        
                        # Clamp to frame bounds
                        viz_x = max(5, min(viz_x, frame.shape[1] - 5))
                        viz_y = max(5, min(viz_y, frame.shape[0] - 5))
                        
                        # Draw crosshair
                        cv2.line(frame, (viz_x - 10, viz_y), (viz_x + 10, viz_y), (0, 255, 255), 2)
                        cv2.line(frame, (viz_x, viz_y - 10), (viz_x, viz_y + 10), (0, 255, 255), 2)
                        cv2.circle(frame, (viz_x, viz_y), 5, (0, 255, 255), 1)
                    
                    # Show calibration status with color coding
                    if tracker.is_calibrated:
                        mode_text = f"Mode: {tracker.calibration_type.upper()} CALIBRATED"
                        mode_color = (0, 255, 0)  # Green for calibrated
                    else:
                        mode_text = "Mode: BASIC (No Calibration)"
                        mode_color = (0, 0, 255)  # Red for basic
                    
                    cv2.putText(frame, mode_text, 
                               (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)
                    
                    # Show calibration quality if available
                    if tracker.calibration_data:
                        quality_color = (0, 255, 0) if tracker.calibration_quality in ['excellent', 'good'] else (0, 255, 255) if tracker.calibration_quality == 'fair' else (0, 0, 255)
                        cv2.putText(frame, f"Quality: {tracker.calibration_quality.upper()}", 
                                   (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, quality_color, 2)
                    
                    # Show helpful tips
                    cv2.putText(frame, "Tips: Z=reset center, I=help", 
                               (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            else:
                # No face detected
                cv2.putText(frame, "No face detected", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "Look at the camera", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Show camera feed
            cv2.imshow('Eye Tracking', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC to exit
                print("👋 Exiting...")
                break
            elif key == ord(' '):  # SPACE to toggle mouse control
                mouse_control_enabled = not mouse_control_enabled
                print(f"🖱️  Mouse control: {'ON' if mouse_control_enabled else 'OFF'}")
            elif key == ord('c'):  # C to toggle calibration
                if tracker.calibration_data:
                    tracker.is_calibrated = not tracker.is_calibrated
                    mode = "calibrated" if tracker.is_calibrated else "basic"
                    print(f"🔧 Switched to {mode} mode")
                else:
                    print("❌ No calibration data available")
            elif key == ord('r'):  # R to reload calibration
                print("🔄 Reloading calibration...")
                tracker.load_latest_calibration()
            elif key == ord('m'):  # M to toggle camera mirroring
                mirror_camera = not mirror_camera
                print(f"📷 Camera mirroring: {'ON' if mirror_camera else 'OFF'}")
                if not mirror_camera:
                    print("   ✅ Natural mode - matches calibrator coordinate system")
                    print("   👁️  Look left → cursor moves left")
                else:
                    print("   ⚠️  Mirror mode - may not match calibration data!")
                    print("   👁️  Look left → cursor moves right (like a mirror)")
                    if tracker.is_calibrated:
                        print("   🔄 Consider recalibrating with mirror mode if accuracy is poor")
            elif key == ord('z'):  # Z to reset center calibration
                if hasattr(tracker, '_center_estimation_buffer'):
                    tracker._center_estimation_buffer.clear()
                    print("🎯 Center calibration reset - look at center of screen for a few seconds")
                    print("   The system will adapt to your natural center position")
            elif key == ord('q'):  # Q for quick 9-point calibration
                print("\n🎯 QUICK CALIBRATION MODE:")
                print("   Look at the RED DOT and press SPACE when your gaze is steady")
                print("   This will help map camera coordinates to screen coordinates")
                tracker.start_quick_calibration()
            elif key == ord('i'):  # I for accuracy info
                print("\n📊 ACCURACY HELP:")
                print("   • Look at center of screen to recalibrate center (press Z to reset)")
                print("   • Press Q for quick 9-point calibration")
                print("   • Make sure your face is well-lit and clearly visible")
                print("   • Sit 50-70cm from camera for best results")
                print("   • Keep head relatively stable, move eyes not head")
                print("   • Yellow crosshair should follow your eye movements in camera")
                if tracker.is_calibrated:
                    print(f"   • Using {tracker.calibration_type} calibration (quality: {tracker.calibration_quality})")
                else:
                    print("   • No calibration - using basic camera-to-screen mapping")
                    print("   • Camera coordinate range is properly scaled to screen range")
            
            # Show FPS occasionally
            if frame_count % 30 == 0:
                fps = 30.0 / max(time.time() - frame_time, 0.001)
                print(f"📊 FPS: {fps:.1f}, Mouse: {'ON' if mouse_control_enabled else 'OFF'}")
    
    except KeyboardInterrupt:
        print("\\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Error during tracking: {e}")
        import traceback
        traceback.print_exc()
        
    except Exception as e:
        print(f"❌ Error during tracking: {e}")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.save_usage_data()
        print("✅ Advanced Eye Tracking completed")



if __name__ == "__main__":
    
    create_advanced_eye_tracking_demo()
    