import React, { useState, useRef } from 'react';
import { Upload, Image as ImageIcon, Camera, AlertCircle, Sparkles } from 'lucide-react';

export default function Home({ onUpload, onError }) {
  const [isDragActive, setIsDragActive] = useState(false);
  const fileInputRef = useRef(null);

  const processFile = (file) => {
    if (!file) return;

    // Validate type locally
    const allowedTypes = ['image/jpeg', 'image/png', 'image/jpg'];
    if (!allowedTypes.includes(file.type)) {
      onError('Unsupported file format. Please upload a high-quality JPEG or PNG image.');
      return;
    }

    // Pass file up to parent
    onUpload(file);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragActive(true);
  };

  const handleDragLeave = () => {
    setIsDragActive(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      processFile(e.target.files[0]);
    }
  };

  const triggerFileSelect = () => {
    fileInputRef.current.click();
  };

  return (
    <div className="max-w-2xl mx-auto px-4 py-6">
      {/* Visual Header */}
      <div className="text-center mb-8">
        <div className="inline-flex items-center gap-1.5 px-3 py-1 bg-teal-50 border border-teal-100 rounded-full text-teal-700 text-xs font-semibold mb-3">
          <Sparkles className="h-3 w-3" />
          Powered by Deep Learning AI
        </div>
        <h1 className="text-3xl md:text-4xl font-extrabold text-slate-800 tracking-tight">
          DermaVision
        </h1>
        <p className="text-slate-500 mt-2 text-base md:text-lg">
          Early detection saves lives. Analysing lesion morphology with precision.
        </p>
      </div>

      {/* Upload Drag & Drop Area */}
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={triggerFileSelect}
        className={`relative border-2 border-dashed rounded-2xl p-8 md:p-12 text-center cursor-pointer transition-all duration-300 flex flex-col items-center justify-center min-h-[300px] ${
          isDragActive
            ? 'border-teal-500 bg-teal-50/50 scale-[1.01] shadow-lg shadow-teal-500/5'
            : 'border-slate-200 bg-white hover:border-teal-400 hover:shadow-md hover:shadow-slate-100'
        }`}
      >
        <input
          type="file"
          ref={fileInputRef}
          onChange={handleFileChange}
          accept=".jpg,.jpeg,.png"
          className="hidden"
        />

        {/* Pulsing upload icon */}
        <div className="h-16 w-16 rounded-full bg-teal-50 border border-teal-100 flex items-center justify-center text-teal-600 mb-4 transition duration-300 group-hover:scale-105">
          <Upload className="h-7 w-7" />
        </div>

        <h3 className="font-bold text-slate-700 text-lg md:text-xl">
          Upload Dermoscopy Photo
        </h3>
        
        <p className="text-slate-400 text-sm mt-2 max-w-sm leading-relaxed">
          Drag and drop your image file here, or <span className="text-teal-600 font-semibold underline decoration-2">browse files</span> from your device.
        </p>

        <div className="flex gap-4 mt-6 text-xs text-slate-500 font-semibold">
          <span className="flex items-center gap-1.5 px-3 py-1.5 bg-slate-50 rounded-lg">
            <ImageIcon className="h-3.5 w-3.5 text-slate-400" /> JPG / PNG
          </span>
          <span className="flex items-center gap-1.5 px-3 py-1.5 bg-slate-50 rounded-lg">
            <Camera className="h-3.5 w-3.5 text-slate-400" /> Close-up
          </span>
        </div>
      </div>

      {/* Guidelines Card */}
      <div className="mt-8 bg-slate-50/50 border border-slate-100 rounded-2xl p-5">
        <h4 className="font-bold text-slate-700 text-sm flex items-center gap-2 mb-3">
          <AlertCircle className="h-4.5 w-4.5 text-teal-600" />
          For best results:
        </h4>
        <ul className="text-xs text-slate-500 space-y-2 leading-relaxed list-disc list-inside">
          <li>Ensure the skin lesion is in the center of the frame.</li>
          <li>Ensure the camera lens is clean and focused (not blurry).</li>
          <li>Use adequate, natural light or macro-flash without glare/reflection.</li>
          <li>Avoid photos with heavy skin markings, hair blocks, or marker ink borders.</li>
        </ul>
      </div>
    </div>
  );
}
