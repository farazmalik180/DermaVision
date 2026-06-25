import React from 'react';
import { Activity } from 'lucide-react';

export default function PulseLoader({ message = "Analysing your image..." }) {
  return (
    <div className="fixed inset-0 z-50 flex flex-col items-center justify-center bg-slate-50/80 backdrop-blur-sm transition-opacity duration-300">
      <div className="relative flex items-center justify-center">
        {/* Pulsing background rings */}
        <div className="absolute h-36 w-36 animate-ping rounded-full bg-teal-100 opacity-60"></div>
        <div className="absolute h-24 w-24 animate-pulse rounded-full bg-teal-50/80 border border-teal-200"></div>
        
        {/* Central icon container */}
        <div className="relative flex h-16 w-16 items-center justify-center rounded-full bg-teal-600 text-white shadow-xl shadow-teal-600/20">
          <Activity className="h-8 w-8 animate-pulse" />
        </div>
      </div>
      
      {/* Loading message */}
      <h3 className="mt-8 text-xl font-bold text-slate-800 animate-breathe">
        {message}
      </h3>
      <p className="mt-2 text-sm text-slate-500">
        Extracting lesion features and generating Grad-CAM activation map...
      </p>
    </div>
  );
}
