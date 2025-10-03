# 🎯 Pure Eye Movement Calibration Guide

## Enhanced Calibration System

The calibration system has been upgraded with the following improvements:

### 🆕 New Features

1. **25-Point Calibration Grid** (upgraded from 9 points)
   - 5x5 grid with additional corner points for better edge accuracy
   - More precise mapping across the entire screen
   - Better performance at screen edges and corners

2. **Gaze Boundary Detection** 
   - Cursor only moves when you're looking at the laptop screen
   - Automatically pauses cursor movement when you look away
   - Prevents accidental cursor movement when distracted

3. **Improved Head Movement Detection**
   - Stricter head movement threshold (12 pixels)
   - Better pure eye movement isolation
   - More accurate calibration data collection

### 📋 Calibration Process

1. **Run the calibrator:**
   ```
   python pure_eye_calibrator.py
   ```

2. **Follow the 29-point sequence:**
   - Look at each target point that appears
   - Keep your head completely still
   - Press SPACE when you're looking directly at the target
   - Wait for data collection (1.5 seconds per point)

3. **Calibration tips:**
   - Sit in your normal working position
   - Ensure good lighting on your face
   - Keep head still throughout the entire process
   - Look precisely at each target center

### 🎮 Using the Enhanced System

1. **Run the advanced tracker:**
   ```
   python advanced_eye_tracker.py
   ```

2. **Controls:**
   - `ESC` - Exit
   - `SPACE` - Toggle mouse control on/off
   - `C` - Toggle between calibrated and basic mapping
   - `B` - Manual blink click
   - `S` - Save session data

3. **Status indicators:**
   - **Cursor: ACTIVE** - Normal operation, cursor follows your gaze
   - **Cursor: PAUSED (looking away)** - You're looking away from screen
   - **Mode: Pure Eye** - Using calibrated eye tracking
   - **Mode: Head+Eye** - Using basic head tracking

### 🔍 Gaze Boundary System

The system automatically detects when you're:
- ✅ Looking at the laptop screen (cursor moves)
- ❌ Looking away from screen (cursor pauses)
- 🎯 Focused vs distracted (based on gaze stability)

This prevents cursor movement when you:
- Look at your phone
- Talk to someone
- Look out the window
- Get distracted

### 📊 Calibration Quality

Good calibration typically achieves:
- **< 50 pixels RMSE**: Excellent accuracy
- **50-100 pixels RMSE**: Good accuracy  
- **> 100 pixels RMSE**: May need recalibration

### 🔧 Troubleshooting

**Poor calibration accuracy:**
- Ensure consistent head position
- Check lighting conditions
- Recalibrate in your actual working environment
- Make sure to look precisely at target centers

**Cursor doesn't move:**
- Press `C` to toggle calibration mode
- Check if you're looking at the screen area
- Verify calibration file was created

**Cursor too sensitive/not sensitive enough:**
- Recalibrate with more careful eye positioning
- Adjust mouse smoothing in the code if needed

### 🎯 Best Practices

1. **Calibration Environment:**
   - Same lighting as normal use
   - Same seating position
   - Same distance from screen

2. **During Use:**
   - Maintain similar head position as calibration
   - Normal eye movements work best
   - Avoid extreme head poses

3. **Recalibration:**
   - When changing seating position significantly
   - When lighting conditions change dramatically
   - If accuracy degrades over time

The enhanced system provides much more precise and natural eye-controlled cursor movement while being smart enough to know when you're actually looking at the screen!