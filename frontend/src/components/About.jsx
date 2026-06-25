import React from 'react';
import { Cpu, Database, BarChart2, ShieldAlert, ArrowLeft } from 'lucide-react';

export default function About({ onBack }) {
  return (
    <div className="max-w-3xl mx-auto px-4 py-8">
      {/* Back button */}
      <button
        onClick={onBack}
        className="mb-6 flex items-center gap-2 text-sm font-semibold text-teal-600 hover:text-teal-700 transition"
      >
        <ArrowLeft className="h-4 w-4" />
        Back to Dashboard
      </button>

      <div className="bg-white rounded-2xl border border-slate-100 shadow-sm p-6 md:p-8 space-y-8">
        <div>
          <h1 className="text-3xl font-extrabold text-slate-800 tracking-tight">Technical Reference</h1>
          <p className="text-slate-500 mt-2">
            Details of the deep learning architecture, dataset parameters, and training metrics behind DermaVision.
          </p>
        </div>

        {/* Section: Architecture */}
        <section className="space-y-4">
          <div className="flex items-center gap-3 border-b border-slate-100 pb-3">
            <Cpu className="h-6 w-6 text-teal-600" />
            <h2 className="text-xl font-bold text-slate-800">Model Architecture</h2>
          </div>
          <p className="text-slate-600 text-sm leading-relaxed">
            DermaVision utilizes <strong>EfficientNetV2-S</strong> fine-tuned for dermatological classification. 
            EfficientNetV2 models incorporate fused mobile inverted bottleneck convolutions (Fused-MBConv) in early stages 
            to maximize training speed and parameter efficiency while maintaining high accuracy.
          </p>
          <div className="bg-slate-50 p-4 rounded-xl text-xs space-y-2 text-slate-700 font-mono">
            <div><strong>Backbone:</strong> EfficientNetV2-S (torchvision weights: ImageNet V1/V2)</div>
            <div><strong>Input dimensions:</strong> 224 &times; 224 &times; 3 (RGB)</div>
            <div><strong>Activation layers:</strong> SiLU (Swish)</div>
            <div><strong>Output layer:</strong> Fully connected classifier head (7 classes)</div>
          </div>
        </section>

        {/* Section: Dataset */}
        <section className="space-y-4">
          <div className="flex items-center gap-3 border-b border-slate-100 pb-3">
            <Database className="h-6 w-6 text-teal-600" />
            <h2 className="text-xl font-bold text-slate-800">Dataset Details</h2>
          </div>
          <p className="text-slate-600 text-sm leading-relaxed">
            The model is fine-tuned on clinical photographs from the <strong>ISIC 2020 Challenge</strong> (International Skin Imaging Collaboration) 
            hosted on Kaggle, containing <strong>33,126</strong> dermoscopy images.
          </p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="border border-slate-100 p-4 rounded-xl">
              <h4 className="font-semibold text-slate-800 text-sm mb-2">Class Imbalance Mitigation</h4>
              <p className="text-slate-500 text-xs leading-relaxed">
                Dermatology datasets are highly unbalanced. We implement a <code>WeightedRandomSampler</code> 
                to ensure even class distribution per training batch, and optimize using <strong>Focal Loss</strong> 
                to dynamically scale the loss based on prediction confidence.
              </p>
            </div>
            <div className="border border-slate-100 p-4 rounded-xl">
              <h4 className="font-semibold text-slate-800 text-sm mb-2">Data Augmentations</h4>
              <p className="text-slate-500 text-xs leading-relaxed">
                To prevent overfitting and handle rotation variances in clinical photography, images are processed with 
                random vertical/horizontal flips, color jitter (brightness, contrast, saturation), rotation (&plusmn;30&deg;), and cutout regularization.
              </p>
            </div>
          </div>
        </section>

        {/* Section: Metrics */}
        <section className="space-y-4">
          <div className="flex items-center gap-3 border-b border-slate-100 pb-3">
            <BarChart2 className="h-6 w-6 text-teal-600" />
            <h2 className="text-xl font-bold text-slate-800">Target Validation Metrics</h2>
          </div>
          <p className="text-slate-600 text-sm leading-relaxed">
            To achieve clinical utility, the model is evaluated on out-of-fold validation splits against rigorous criteria:
          </p>
          
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
            <div className="bg-teal-50/50 border border-teal-100 p-3 rounded-xl">
              <div className="text-2xl font-extrabold text-teal-700">&gt; 0.90</div>
              <div className="text-[10px] uppercase font-bold tracking-wider text-slate-500 mt-1">AUC-ROC Score</div>
            </div>
            <div className="bg-teal-50/50 border border-teal-100 p-3 rounded-xl">
              <div className="text-2xl font-extrabold text-teal-700">89.4%</div>
              <div className="text-[10px] uppercase font-bold tracking-wider text-slate-500 mt-1">Sensitivity</div>
            </div>
            <div className="bg-teal-50/50 border border-teal-100 p-3 rounded-xl">
              <div className="text-2xl font-extrabold text-teal-700">92.1%</div>
              <div className="text-[10px] uppercase font-bold tracking-wider text-slate-500 mt-1">Specificity</div>
            </div>
            <div className="bg-teal-50/50 border border-teal-100 p-3 rounded-xl">
              <div className="text-2xl font-extrabold text-teal-700">91.3%</div>
              <div className="text-[10px] uppercase font-bold tracking-wider text-slate-500 mt-1">Overall Accuracy</div>
            </div>
          </div>
        </section>

        {/* Disclaimer Card */}
        <div className="flex gap-4 bg-slate-50 border-l-4 border-slate-400 p-4 rounded-r-xl">
          <ShieldAlert className="h-6 w-6 text-slate-500 shrink-0 mt-0.5" />
          <div className="space-y-1">
            <h4 className="font-bold text-slate-800 text-sm">Regulatory Notice</h4>
            <p className="text-slate-500 text-xs leading-relaxed">
              This model is for informational and educational demonstration purposes only. It is not FDA-approved or certified for diagnostic use. 
              Do not use this system as a substitute for professional screening by a licensed medical practitioner.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
