import React from 'react';
import { AlertTriangle, ShieldCheck, RefreshCw, Eye } from 'lucide-react';

export default function Results({ originalImage, result, onReset }) {
  const { label, confidence, risk_level, description, gradcam_image_base64 } = result;

  const isLow = risk_level === 'Low';
  const isModerate = risk_level === 'Moderate';
  const isHigh = risk_level === 'High';

  // Determine badge colors based on risk
  const riskBadgeClasses = isHigh
    ? 'bg-red-50 text-red-700 border-red-200'
    : isModerate
      ? 'bg-amber-50 text-amber-700 border-amber-200'
      : 'bg-emerald-50 text-emerald-700 border-emerald-200';

  const riskDotColor = isHigh ? 'bg-red-600' : isModerate ? 'bg-amber-600' : 'bg-emerald-600';
  const confidenceColor = isHigh ? 'bg-red-600' : isModerate ? 'bg-amber-500' : 'bg-teal-600';

  // Read URL for local preview
  const originalUrl = originalImage ? URL.createObjectURL(originalImage) : null;

  return (
    <div className="max-w-4xl mx-auto px-4 py-6 space-y-6">
      <div className="bg-white border border-slate-100 rounded-2xl p-6 md:p-8 shadow-sm">
        
        {/* Title Block */}
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-100 pb-5">
          <div>
            <span className="text-xs font-bold text-slate-400 uppercase tracking-widest">Diagnostic Report</span>
            <h2 className="text-2xl font-extrabold text-slate-800 mt-1">{label}</h2>
          </div>
          
          <div className="flex items-center gap-2">
            <span className={`inline-flex items-center gap-2 px-4 py-1.5 rounded-full border text-sm font-bold shadow-xs ${riskBadgeClasses}`}>
              <span className={`h-2.5 w-2.5 rounded-full ${riskDotColor}`}></span>
              {risk_level} Risk Level
            </span>
          </div>
        </div>

        {/* Comparison Images */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 my-6">
          <div className="space-y-2">
            <h4 className="text-xs font-bold text-slate-500 uppercase tracking-wider flex items-center gap-1.5">
              <Eye className="h-3.5 w-3.5" /> Original Photo
            </h4>
            <div className="aspect-square bg-slate-50 rounded-xl border border-slate-100 overflow-hidden flex items-center justify-center relative">
              {originalUrl ? (
                <img 
                  src={originalUrl} 
                  alt="Original lesion" 
                  className="w-full h-full object-cover" 
                />
              ) : (
                <div className="text-slate-400 text-sm">Preview Unavailable</div>
              )}
            </div>
          </div>

          <div className="space-y-2">
            <h4 className="text-xs font-bold text-slate-500 uppercase tracking-wider flex items-center gap-1.5">
              <Eye className="h-3.5 w-3.5 text-rose-500 animate-pulse" /> Grad-CAM Heatmap
            </h4>
            <div className="aspect-square bg-slate-50 rounded-xl border border-slate-100 overflow-hidden flex items-center justify-center relative">
              {gradcam_image_base64 ? (
                <img 
                  src={gradcam_image_base64} 
                  alt="Gradcam heatmap" 
                  className="w-full h-full object-cover" 
                />
              ) : (
                <div className="text-slate-400 text-sm">Grad-CAM overlay failed to generate</div>
              )}
            </div>
          </div>
        </div>

        {/* Stats & Confidence Section */}
        <div className="space-y-6 bg-slate-50/50 border border-slate-100 p-5 rounded-2xl">
          <div>
            <div className="flex justify-between items-center text-sm font-bold text-slate-700 mb-2">
              <span>Classifier Confidence</span>
              <span className="text-base text-slate-900 font-extrabold">{(confidence * 100).toFixed(1)}%</span>
            </div>
            
            {/* Custom Progress Bar */}
            <div className="h-3 w-full bg-slate-200/70 rounded-full overflow-hidden">
              <div 
                className={`h-full rounded-full transition-all duration-1000 ${confidenceColor}`}
                style={{ width: `${confidence * 100}%` }}
              ></div>
            </div>
          </div>

          {/* Diagnosis description */}
          {description && (
            <div className="text-slate-600 text-sm leading-relaxed border-t border-slate-200/40 pt-4">
              <strong className="text-slate-700">Description: </strong> {description}
            </div>
          )}
        </div>

        {/* Disclaimer Banner */}
        <div className="mt-8 flex gap-4 bg-slate-50 border-l-4 border-slate-400 p-4 rounded-r-xl">
          <AlertTriangle className="h-5.5 w-5.5 text-slate-500 shrink-0 mt-0.5" />
          <div className="space-y-1">
            <h5 className="font-bold text-slate-800 text-sm">Medical Disclaimer</h5>
            <p className="text-slate-500 text-xs leading-relaxed">
              This report is generated by a computer vision model for demonstrative evaluation. 
              It is not a medical diagnosis. Always seek clinical consult from a certified dermatologist for any skin condition or lesion.
            </p>
          </div>
        </div>

        {/* Scan Another Button */}
        <div className="mt-8 flex justify-center">
          <button
            onClick={onReset}
            className="flex items-center gap-2 px-6 py-3 bg-teal-600 hover:bg-teal-700 text-white font-bold rounded-xl shadow-lg shadow-teal-600/10 hover:shadow-xl hover:shadow-teal-700/20 active:scale-[0.98] transition cursor-pointer"
          >
            <RefreshCw className="h-4.5 w-4.5" />
            Scan Another Lesion
          </button>
        </div>

      </div>
    </div>
  );
}
