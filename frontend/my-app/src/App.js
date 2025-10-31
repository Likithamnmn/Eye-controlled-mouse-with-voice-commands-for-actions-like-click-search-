import React, { useState } from 'react';
import StartPage from './components/StartPage';
import EyeVoiceWidget from './components/EyeVoiceWidget';
import { ArrowLeft } from 'lucide-react';
import { Button } from './components/ui/button';

export default function App() {
  const [showStartPage, setShowStartPage] = useState(true);

  if (showStartPage) {
    return <StartPage onGetStarted={() => setShowStartPage(false)} />;
  }

  return (
    <div className="min-h-screen flex items-center justify-center p-8 bg-gradient-purple bg-fixed">
      {/* Back button */}
      <Button
        onClick={() => setShowStartPage(true)}
        className="absolute top-8 left-8 z-20 bg-white/20 backdrop-blur-sm text-white border-white/30 hover:bg-white/30 transition-all duration-200"
        variant="outline"
      >
        <ArrowLeft className="w-4 h-4 mr-2" />
        Back to Home
      </Button>

      {/* Floating sticky notes */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-10 left-10 w-16 h-16 rounded-lg rotate-12 opacity-20 bg-yellow-200"></div>
        <div className="absolute top-32 right-20 w-12 h-12 rounded-lg -rotate-6 opacity-20 bg-pink-200"></div>
        <div className="absolute bottom-20 left-32 w-20 h-20 rounded-lg rotate-45 opacity-20 bg-blue-200"></div>
        <div className="absolute bottom-32 right-10 w-14 h-14 rounded-lg -rotate-12 opacity-20 bg-green-200"></div>
      </div>

      {/* Main content */}
      <div className="relative z-10 w-full max-w-2xl">
        <div className="bg-cork-board p-8 rounded-2xl shadow-2xl border-4 border-amber-300">
          <EyeVoiceWidget onBackHome={() => setShowStartPage(true)} />
        </div>
      </div>
    </div>
  );
}
