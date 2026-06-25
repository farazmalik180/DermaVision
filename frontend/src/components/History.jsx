import React from 'react';
import { Clock, Trash2, ShieldCheck } from 'lucide-react';

export default function History({ history, onClear }) {
  return (
    <div className="bg-white border border-slate-100 rounded-2xl p-6 shadow-sm">
      <div className="flex items-center justify-between mb-4 border-b border-slate-50 pb-3">
        <div className="flex items-center gap-2">
          <Clock className="h-5 w-5 text-slate-400" />
          <h3 className="font-bold text-slate-800 text-base">Recent Scans</h3>
        </div>
        {history && history.length > 0 && (
          <button
            onClick={onClear}
            className="text-xs font-semibold text-slate-400 hover:text-red-500 flex items-center gap-1 transition cursor-pointer"
            title="Clear all local history"
          >
            <Trash2 className="h-3.5 w-3.5" />
            Clear
          </button>
        )}
      </div>

      {(!history || history.length === 0) ? (
        <div className="py-8 text-center text-slate-400 text-xs">
          No recent scans. Upload an image to start analysis.
        </div>
      ) : (
        <div className="space-y-3">
          {history.map((scan, idx) => {
            const isHigh = scan.risk === 'High';
            const isModerate = scan.risk === 'Moderate';
            const riskColor = isHigh 
              ? 'text-red-600 bg-red-50 border-red-100' 
              : isModerate 
                ? 'text-amber-600 bg-amber-50 border-amber-100' 
                : 'text-emerald-600 bg-emerald-50 border-emerald-100';

            return (
              <div 
                key={idx} 
                className="p-3 bg-slate-50/50 hover:bg-slate-50 border border-slate-100 rounded-xl flex flex-col gap-1 transition"
              >
                <div className="flex justify-between items-center text-[10px] text-slate-400">
                  <span className="font-medium">Scan #{history.length - idx}</span>
                  <span>{scan.timestamp}</span>
                </div>
                
                <h4 className="font-semibold text-slate-800 text-sm mt-0.5 truncate">{scan.label}</h4>
                
                <div className="flex items-center justify-between mt-1 text-xs">
                  <span className={`px-2 py-0.5 rounded-full border text-[10px] font-bold ${riskColor}`}>
                    {scan.risk} Risk
                  </span>
                  <span className="font-semibold text-slate-600">
                    {(scan.confidence * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
