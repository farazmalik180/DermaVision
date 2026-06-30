import React, { useState, useEffect, useRef } from 'react';
import { AlertTriangle, ShieldCheck, RefreshCw, Eye, Download, Save } from 'lucide-react';
import html2canvas from 'html2canvas-pro';
import { jsPDF } from 'jspdf';
import { getProfiles, addScan } from '../db';

export default function Results({ originalImage, result, onReset }) {
  const { label, confidence, risk_level, description, gradcam_image_base64 } = result;

  const [imageBase64, setImageBase64] = useState(null);
  const [profiles, setProfiles] = useState([]);
  const [selectedProfileId, setSelectedProfileId] = useState('');
  const [saveStatus, setSaveStatus] = useState('');
  const reportRef = useRef(null);

  useEffect(() => {
    if (originalImage) {
      const reader = new FileReader();
      reader.onloadend = () => setImageBase64(reader.result);
      reader.readAsDataURL(originalImage);
    }
    getProfiles().then(p => setProfiles(p));
  }, [originalImage]);

  const handleDownloadPDF = async () => {
    const element = reportRef.current;
    
    try {
      const canvas = await html2canvas(element, { scale: 2, useCORS: true, logging: true });
      const imgData = canvas.toDataURL('image/jpeg', 0.98);
      
      const pdf = new jsPDF({
        unit: 'mm',
        format: 'a4',
        orientation: 'portrait'
      });
      
      const pdfWidth = pdf.internal.pageSize.getWidth();
      const pdfHeight = (canvas.height * pdfWidth) / canvas.width;
      
      pdf.addImage(imgData, 'JPEG', 0, 0, pdfWidth, pdfHeight);
      
      const safeLabel = label ? label.replace(/\s+/g, '_') : 'Scan';
      pdf.save(`DermaVision_Report_${safeLabel}.pdf`);
    } catch (error) {
      console.error("PDF generation failed:", error);
    }
  };

  const handleSaveToProfile = async () => {
    if (!selectedProfileId || !imageBase64) return;
    try {
      await addScan(selectedProfileId, imageBase64, result);
      setSaveStatus('Saved successfully!');
      setTimeout(() => setSaveStatus(''), 3000);
    } catch (err) {
      setSaveStatus('Failed to save.');
    }
  };

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
        
        <div ref={reportRef}>
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
        </div>

        {/* Action Buttons */}
        <div className="mt-8 flex flex-col md:flex-row items-center justify-between gap-4 border-t border-slate-100 pt-6">
          <div className="flex items-center gap-2 w-full md:w-auto">
            <select 
              value={selectedProfileId}
              onChange={(e) => setSelectedProfileId(e.target.value)}
              className="text-sm border border-slate-200 rounded-lg px-3 py-2 outline-none focus:border-teal-500 bg-slate-50 flex-1"
            >
              <option value="">-- Select Profile --</option>
              {profiles.map(p => (
                <option key={p.id} value={p.id}>{p.name}</option>
              ))}
            </select>
            <button
              onClick={handleSaveToProfile}
              disabled={!selectedProfileId}
              className="flex items-center gap-2 px-4 py-2 bg-slate-800 disabled:bg-slate-300 hover:bg-slate-900 text-white font-bold rounded-lg transition cursor-pointer"
            >
              <Save className="h-4 w-4" />
              Save
            </button>
            {saveStatus && <span className="text-xs font-bold text-teal-600">{saveStatus}</span>}
          </div>

          <div className="flex items-center gap-3 w-full md:w-auto">
            <button
              onClick={handleDownloadPDF}
              className="flex-1 md:flex-none flex items-center justify-center gap-2 px-4 py-2 border border-slate-200 text-slate-700 font-bold rounded-lg hover:bg-slate-50 transition cursor-pointer"
            >
              <Download className="h-4.5 w-4.5" />
              PDF Report
            </button>
            <button
              onClick={onReset}
              className="flex-1 md:flex-none flex items-center justify-center gap-2 px-6 py-2 bg-teal-600 hover:bg-teal-700 text-white font-bold rounded-lg shadow-lg shadow-teal-600/10 hover:shadow-xl hover:shadow-teal-700/20 active:scale-[0.98] transition cursor-pointer"
            >
              <RefreshCw className="h-4.5 w-4.5" />
              New Scan
            </button>
          </div>
        </div>

      </div>
    </div>
  );
}
