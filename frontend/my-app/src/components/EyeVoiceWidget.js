import React, { useState } from 'react';
import { Eye, Mic, Hand, Settings, ArrowLeft } from 'lucide-react';
import Switch from './ui/Switch'; // Your specific toggle component

export default function EyeVoiceWidget({ onBackHome }) {
  // Main toggles
  const [eyeEnabled, setEyeEnabled] = useState(false);
  const [voiceEnabled, setVoiceEnabled] = useState(false);
  const [gestureEnabled, setGestureEnabled] = useState(false);

  // Sub-options
  const [eyeOptions, setEyeOptions] = useState([false, false]);
  const [voiceOptions, setVoiceOptions] = useState([false, false, false]);
  const [gestureOptions, setGestureOptions] = useState([false, false, false]);

  const eyeLabels = ['Motion Detection', 'Blink Detection'];
  const voiceLabels = ['Scrolling', 'Speech Recognition', 'Volume Control'];
  const gestureLabels = ['Swipe Gestures', 'Pinch Controls', 'Palm Recognition'];

  const toggleOption = (index, arr, setFn) => {
    const copy = [...arr];
    copy[index] = !copy[index];
    setFn(copy);
  };

  const SubOptions = ({ options, labels, setFn }) => (
    <div className="ml-6 mt-3 space-y-2">
      {options.map((opt, i) => (
        <div
          key={i}
          className="flex items-center justify-between bg-gray-800/30 px-4 py-2 rounded-md"
        >
          <span className="text-gray-200">{labels[i]}</span>
          <Switch
            checked={opt}
            onCheckedChange={(val) => toggleOption(i, options, setFn)}
          />
        </div>
      ))}
    </div>
  );

  return (
    <div className="max-w-lg mx-auto p-6 space-y-6">
      {/* Back button */}
      <button
        onClick={onBackHome}
        className="flex items-center gap-2 px-4 py-2 bg-gray-800 hover:bg-gray-700 text-white rounded-lg transition mb-4"
      >
        <ArrowLeft className="w-5 h-5" />
        Back to Home
      </button>

      {/* Header */}
      <div className="text-center mb-6">
        <div className="flex items-center justify-center gap-2 mb-1">
          <Settings className="w-6 h-6 text-amber-500" />
          <h1 className="text-2xl font-semibold text-black-100">Accessibility Settings</h1>
        </div>
        <p className="text-sm text-black-400">Configure your accessibility preferences</p>
      </div>

      {/* Eye Controls */}
      <div className="bg-yellow-700/20 rounded-lg p-4 border-l-4 border-yellow-400 shadow-lg">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Eye className="w-5 h-5 text-yellow-600" />
            <span className="font-medium text-yellow-900">Eye Controls</span>
          </div>
          <Switch
            checked={eyeEnabled}
            onCheckedChange={(val) => setEyeEnabled(val)}
          />
        </div>
        {eyeEnabled && <SubOptions options={eyeOptions} labels={eyeLabels} setFn={setEyeOptions} />}
      </div>

      {/* Voice Controls */}
      <div className="bg-pink-700/20 rounded-lg p-4 border-l-4 border-pink-400 shadow-lg">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Mic className="w-5 h-5 text-pink-600" />
            <span className="font-medium text-pink-900">Voice Controls</span>
          </div>
          <Switch
            checked={voiceEnabled}
            onCheckedChange={(val) => setVoiceEnabled(val)}
          />
        </div>
        {voiceEnabled && <SubOptions options={voiceOptions} labels={voiceLabels} setFn={setVoiceOptions} />}
      </div>

      {/* Hand Gestures */}
      <div className="bg-blue-700/20 rounded-lg p-4 border-l-4 border-blue-400 shadow-lg">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Hand className="w-5 h-5 text-blue-600" />
            <span className="font-medium text-blue-900">Hand Gestures</span>
          </div>
          <Switch
            checked={gestureEnabled}
            onCheckedChange={(val) => setGestureEnabled(val)}
          />
        </div>
        {gestureEnabled && <SubOptions options={gestureOptions} labels={gestureLabels} setFn={setGestureOptions} />}
      </div>
    </div>
  );
}
