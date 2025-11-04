import React, { useState } from "react";
import { Eye, Mic, Hand, Settings, ArrowLeft } from "lucide-react";
import Switch from "./ui/Switch";
import API from "../api";

export default function EyeVoiceWidget({ onBackHome }) {
  const [eyeEnabled, setEyeEnabled] = useState(false);
  const [voiceEnabled, setVoiceEnabled] = useState(false);
  const [pinchEnabled, setPinchEnabled] = useState(false);
  const [palmEnabled, setPalmEnabled] = useState(false);
  const [loading, setLoading] = useState({
    eye: false,
    voice: false,
    pinch: false,
    palm: false,
  });

  /**
   * Toggle function for eye and voice controls
   * - When turning ON: Only updates state on success, reverts on error
   * - When turning OFF: Always allows toggle off (even if API fails)
   */
  const toggle = async (key, fn, setState, val, label) => {
    // Prevent double-clicks while loading
    if (loading[key]) {
      console.log(`${label} is already processing...`);
      return;
    }

    // Store previous state to revert on error ONLY when starting
    const previousState = key === "eye" ? eyeEnabled : 
                         key === "voice" ? voiceEnabled : 
                         key === "pinch" ? pinchEnabled : palmEnabled;
    
    setLoading(prev => ({ ...prev, [key]: true }));
    
    try {
      const res = await fn();
      
      // Check for errors
      if (res?.error) {
        console.error(`${label} error:`, res.error);
        // Only revert on error if we were trying to START
        // If stopping, always allow toggle off even on error
        if (val) {
          setState(previousState); // Revert to previous state on start error
          if (res.status === 403) {
            alert(`${label} is disabled: ${res.error}`);
          } else {
            alert(`Failed to start ${label.toLowerCase()}: ${res.error}`);
          }
        } else {
          // Stopping - always allow toggle off
          setState(false);
          console.log(`${label} stopped`);
        }
      } else {
        // Success - update state to new value
        setState(val);
        console.log(`${label} ${val ? 'started' : 'stopped'}:`, res.status || "success");
      }
    } catch (e) {
      console.error(`${label} failed:`, e);
      // Only revert on error if we were trying to START
      // If stopping, always allow toggle off
      if (val) {
        setState(previousState); // Revert on exception when starting
        alert(`Failed to start ${label.toLowerCase()}`);
      } else {
        // Stopping - always allow toggle off even on network error
        setState(false);
        console.log(`${label} stopped (network error ignored)`);
      }
    } finally {
      setLoading(prev => ({ ...prev, [key]: false }));
    }
  };

  /**
   * Handle gesture controls (pinch and palm)
   * Combines both gestures into appropriate API calls
   */
  const handleGestures = async (pinch, palm) => {
    try {
      if (pinch && palm) {
        await API.system.both_start();
      } else if (pinch) {
        await API.system.volume_start();
      } else if (palm) {
        await API.system.screenshot_start();
      } else {
        await API.system.both_stop();
      }
    } catch (e) {
      console.error("Gesture control failed:", e);
      // Revert UI if the request failed
      setPinchEnabled(false);
      setPalmEnabled(false);
      alert("Failed to update gesture controls.");
    }
  };

  /**
   * Toggle gesture control with loading state
   */
  const toggleGesture = async (gesture, val, otherGesture) => {
    const key = gesture === "pinch" ? "pinch" : "palm";
    const setState = gesture === "pinch" ? setPinchEnabled : setPalmEnabled;
    const currentState = gesture === "pinch" ? pinchEnabled : palmEnabled;
    
    if (loading.pinch || loading.palm) return;
    
    setLoading(prev => ({ ...prev, [key]: true }));
    const prevState = currentState;
    setState(val);
    
    try {
      const otherVal = gesture === "pinch" ? palmEnabled : pinchEnabled;
      await handleGestures(
        gesture === "pinch" ? val : pinchEnabled,
        gesture === "palm" ? val : palmEnabled
      );
    } catch (e) {
      setState(prevState); // Revert on error
    } finally {
      setLoading(prev => ({ ...prev, [key]: false }));
    }
  };

  return (
    <div className="max-w-lg mx-auto p-6 space-y-6">
      {/* Back Button */}
      <button
        onClick={onBackHome}
        className="flex items-center gap-2 px-4 py-2 bg-gray-800 hover:bg-gray-700 text-white rounded-lg transition mb-4"
      >
        <ArrowLeft className="w-5 h-5" /> Back to Home
      </button>

      {/* Header */}
      <div className="text-center mb-6">
        <div className="flex items-center justify-center gap-2 mb-1">
          <Settings className="w-6 h-6 text-amber-500" />
          <h1 className="text-2xl font-semibold text-black">Accessibility Settings</h1>
        </div>
      </div>

      {/* 👁 Eye Control */}
      <div className="bg-yellow-100 rounded-lg p-4 border-l-4 border-yellow-500 shadow">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Eye className="w-5 h-5 text-yellow-600" />
            <span className="font-medium">Eye Control</span>
            {loading.eye && (
              <span className="text-xs text-yellow-700 ml-2 animate-pulse">Processing...</span>
            )}
          </div>
          <Switch
            disabled={loading.eye}
            checked={eyeEnabled}
            onCheckedChange={(val) =>
              toggle("eye", val ? API.eye.start : API.eye.stop, setEyeEnabled, val, "Eye Control")
            }
          />
        </div>
        {eyeEnabled && !loading.eye && (
          <p className="text-xs text-yellow-700 mt-2">Eye tracking is active</p>
        )}
      </div>

      {/* 🎙 Voice Control */}
      <div className="bg-pink-100 rounded-lg p-4 border-l-4 border-pink-500 shadow">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Mic className="w-5 h-5 text-pink-600" />
            <span className="font-medium">Voice Control</span>
            {loading.voice && (
              <span className="text-xs text-pink-700 ml-2 animate-pulse">Processing...</span>
            )}
          </div>
          <Switch
            disabled={loading.voice}
            checked={voiceEnabled}
            onCheckedChange={(val) =>
              toggle("voice", val ? API.voice.start : API.voice.stop, setVoiceEnabled, val, "Voice Control")
            }
          />
        </div>
        {voiceEnabled && !loading.voice && (
          <p className="text-xs text-pink-700 mt-2">Voice commands are active</p>
        )}
      </div>

      {/* ✋ Hand Gestures */}
      <div className="bg-blue-100 rounded-lg p-4 border-l-4 border-blue-500 shadow space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Hand className="w-5 h-5 text-blue-600" />
            <span className="font-medium">Pinch Control</span>
            {loading.pinch && (
              <span className="text-xs text-blue-700 ml-2 animate-pulse">Processing...</span>
            )}
          </div>
          <Switch
            disabled={loading.pinch || loading.palm}
            checked={pinchEnabled}
            onCheckedChange={(val) => toggleGesture("pinch", val, palmEnabled)}
          />
        </div>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Hand className="w-5 h-5 text-blue-600" />
            <span className="font-medium">Palm Recognition</span>
            {loading.palm && (
              <span className="text-xs text-blue-700 ml-2 animate-pulse">Processing...</span>
            )}
          </div>
          <Switch
            disabled={loading.pinch || loading.palm}
            checked={palmEnabled}
            onCheckedChange={(val) => toggleGesture("palm", val, pinchEnabled)}
          />
        </div>
        {(pinchEnabled || palmEnabled) && !loading.pinch && !loading.palm && (
          <p className="text-xs text-blue-700 mt-2">
            {pinchEnabled && palmEnabled
              ? "Volume and screenshot controls active"
              : pinchEnabled
              ? "Volume control active"
              : "Screenshot control active"}
          </p>
        )}
      </div>
    </div>
  );
}