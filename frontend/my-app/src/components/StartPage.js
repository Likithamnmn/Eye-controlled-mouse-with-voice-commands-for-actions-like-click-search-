import React from "react";
import {
  Eye,
  Mic,
  ArrowRight,
  Hand,
} from "lucide-react";
import { Button } from "./ui/button";

interface StartPageProps {
  onGetStarted: () => void;
}

export default function StartPage({ onGetStarted }: StartPageProps) {
  return (
    <div className="min-h-screen flex items-center justify-center p-8 relative overflow-hidden bg-gradient-purple bg-fixed">
      {/* Animated background elements */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-20 left-20 w-32 h-32 rounded-full bg-gradient-to-r from-yellow-200/30 to-transparent animate-float"></div>
        <div className="absolute top-40 right-32 w-24 h-24 rounded-full bg-gradient-to-r from-pink-200/30 to-transparent animate-float-reverse"></div>
        <div className="absolute bottom-32 left-40 w-40 h-40 rounded-full bg-gradient-to-r from-blue-200/30 to-transparent animate-float-slow"></div>
      </div>

      <div className="relative z-10 text-center max-w-4xl mx-auto">
        {/* 🔥 Main Title (black font) */}
        <div className="mb-8">
          <h1 className="text-5xl md:text-6xl font-bold mb-4 leading-tight text-black drop-shadow-lg">
            EYE TRACKING
          </h1>
          <div className="flex items-center justify-center gap-4 mb-4">
            <div className="h-1 w-16 bg-gradient-to-r from-yellow-400 to-pink-400 rounded-full"></div>
            <span className="text-2xl text-black font-light drop-shadow-sm">
              &
            </span>
            <div className="h-1 w-16 bg-gradient-to-r from-pink-400 to-blue-400 rounded-full"></div>
          </div>
          <h2 className="text-4xl md:text-5xl font-bold leading-tight text-black drop-shadow-lg">
            VOICE ACTIVATED
          </h2>
          <h3 className="text-3xl md:text-4xl font-light text-black mt-2 drop-shadow-md">
            INTERFACE
          </h3>
        </div>

        {/* Subtitle */}
        <p className="text-xl text-black mb-12 max-w-2xl mx-auto leading-relaxed drop-shadow-sm">
          Experience the future of accessibility with our
          intuitive eye tracking, voice control, and hand
          gesture system. Configure your perfect setup with our
          easy-to-use interface.
        </p>

        {/* Feature highlights (Eye, Voice, Hand) */}
        <div className="grid md:grid-cols-3 gap-8 mb-12">
          <div className="text-center">
            <div className="w-16 h-16 bg-yellow-400/20 rounded-2xl flex items-center justify-center mx-auto mb-4 backdrop-blur-sm">
              <Eye className="w-8 h-8 text-yellow-600" />
            </div>
            <h4 className="text-lg font-semibold text-black mb-2">
              Eye Tracking
            </h4>
            <p className="text-black/80">
              Precise gaze control and motion detection
            </p>
          </div>
          <div className="text-center">
            <div className="w-16 h-16 bg-pink-400/20 rounded-2xl flex items-center justify-center mx-auto mb-4 backdrop-blur-sm">
              <Mic className="w-8 h-8 text-pink-600" />
            </div>
            <h4 className="text-lg font-semibold text-black mb-2">
              Voice Control
            </h4>
            <p className="text-black/80">
              Advanced speech recognition and commands
            </p>
          </div>
          <div className="text-center">
            <div className="w-16 h-16 bg-blue-400/20 rounded-2xl flex items-center justify-center mx-auto mb-4 backdrop-blur-sm">
              <Hand className="w-8 h-8 text-blue-600" />
            </div>
            <h4 className="text-lg font-semibold text-black mb-2">
              Hand Gestures
            </h4>
            <p className="text-black/80">
              Intuitive gesture recognition and control
            </p>
          </div>
        </div>

        {/* CTA Button */}
        <Button
          onClick={onGetStarted}
          className="group px-12 py-6 text-xl font-semibold rounded-2xl shadow-2xl hover:shadow-3xl transform hover:scale-105 transition-all duration-300 border-0 bg-gradient-to-r from-yellow-400 via-pink-400 to-blue-400 text-white ring-2 ring-white/10"
        >
          <span className="flex items-center gap-3">
            Get Started
            <ArrowRight className="w-6 h-6 group-hover:translate-x-2 transition-transform duration-300" />
          </span>
        </Button>
      </div>
    </div>
  );
}
