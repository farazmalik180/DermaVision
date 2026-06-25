import React, { useState, useEffect } from 'react';
import { Activity, ShieldAlert, BookOpen, UploadCloud, AlertCircle } from 'lucide-react';

// Components
import Home from './components/Home';
import Results from './components/Results';
import About from './components/About';
import History from './components/History';
import PulseLoader from './components/PulseLoader';

export default function App() {
  const [activeTab, setActiveTab] = useState('scan'); // 'scan', 'results', 'about'
  const [selectedFile, setSelectedFile] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [loadingMessage, setLoadingMessage] = useState('Analysing your image...');
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [history, setHistory] = useState([]);

  const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

  // Load history from localStorage on mount
  useEffect(() => {
    try {
      const storedHistory = localStorage.getItem('dermavision_history');
      if (storedHistory) {
        setHistory(JSON.parse(storedHistory));
      }
    } catch (e) {
      console.error('Failed to load history from local storage:', e);
    }
  }, []);

  const handleClearHistory = () => {
    setHistory([]);
    localStorage.removeItem('dermavision_history');
  };

  const handleUpload = async (file) => {
    setSelectedFile(file);
    setIsLoading(true);
    setError(null);
    setLoadingMessage('Analysing your image...');

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        // Handle server/validation errors (e.g. blurry image)
        throw new Error(data.detail || 'An unexpected error occurred during prediction.');
      }

      // Successful prediction
      setResult(data);

      // Add to history list
      const timestamp = new Date().toLocaleTimeString([], { 
        hour: '2-digit', 
        minute: '2-digit', 
        second: '2-digit' 
      });
      const newScan = {
        label: data.label,
        risk: data.risk_level,
        confidence: data.confidence,
        timestamp: timestamp
      };

      const updatedHistory = [newScan, ...history].slice(0, 5);
      setHistory(updatedHistory);
      localStorage.setItem('dermavision_history', JSON.stringify(updatedHistory));

      // Navigate to results screen
      setActiveTab('results');
    } catch (err) {
      console.error('Analysis error:', err);
      setError(err.message || 'Failed to connect to the analysis server. Please make sure the backend server is running.');
      setSelectedFile(null);
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setResult(null);
    setSelectedFile(null);
    setError(null);
    setActiveTab('scan');
  };

  return (
    <div className="min-h-screen bg-slate-50 flex flex-col font-sans text-slate-800">
      {/* Pulse Loader Overlay */}
      {isLoading && <PulseLoader message={loadingMessage} />}

      {/* Premium Top Navigation Bar */}
      <header className="bg-white border-b border-slate-100 sticky top-0 z-40">
        <div className="max-w-6xl mx-auto px-4 h-16 flex items-center justify-between">
          <div 
            onClick={handleReset}
            className="flex items-center gap-2 cursor-pointer group"
          >
            <div className="h-9 w-9 rounded-lg bg-teal-600 text-white flex items-center justify-center shadow-md shadow-teal-600/10 group-hover:scale-105 transition">
              <Activity className="h-5 w-5" />
            </div>
            <span className="font-extrabold text-xl tracking-tight text-slate-900">
              Derma<span className="text-teal-600">Vision</span>
            </span>
          </div>

          <nav className="flex items-center gap-1 md:gap-2">
            <button
              onClick={() => { handleReset(); setActiveTab('scan'); }}
              className={`px-4 py-2 text-sm font-semibold rounded-lg transition cursor-pointer ${
                activeTab === 'scan' || activeTab === 'results'
                  ? 'bg-teal-50 text-teal-700'
                  : 'text-slate-500 hover:text-slate-800 hover:bg-slate-50'
              }`}
            >
              Scan
            </button>
            <button
              onClick={() => setActiveTab('about')}
              className={`px-4 py-2 text-sm font-semibold rounded-lg transition cursor-pointer flex items-center gap-1.5 ${
                activeTab === 'about'
                  ? 'bg-teal-50 text-teal-700'
                  : 'text-slate-500 hover:text-slate-800 hover:bg-slate-50'
              }`}
            >
              <BookOpen className="h-4.5 w-4.5" />
              Technical Info
            </button>
          </nav>
        </div>
      </header>

      {/* Main Container */}
      <main className="flex-grow max-w-6xl w-full mx-auto px-4 py-6">
        
        {/* Error notification banner */}
        {error && (
          <div className="max-w-2xl mx-auto mb-6 bg-red-50 border border-red-200 text-red-700 rounded-xl p-4 flex gap-3 items-start animate-pulse">
            <AlertCircle className="h-5 w-5 shrink-0 mt-0.5" />
            <div className="space-y-1">
              <h5 className="font-bold text-sm">Analysis Failed</h5>
              <p className="text-xs leading-relaxed">{error}</p>
            </div>
            <button 
              onClick={() => setError(null)}
              className="text-xs font-semibold text-red-500 hover:text-red-700 ml-auto cursor-pointer"
            >
              Dismiss
            </button>
          </div>
        )}

        {/* View Switcher Router */}
        {activeTab === 'about' && (
          <About onBack={handleReset} />
        )}

        {activeTab === 'results' && result && (
          <Results 
            originalImage={selectedFile} 
            result={result} 
            onReset={handleReset} 
          />
        )}

        {activeTab === 'scan' && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 items-start">
            <div className="lg:col-span-2">
              <Home 
                onUpload={handleUpload} 
                onError={(errMessage) => setError(errMessage)} 
              />
            </div>
            
            {/* Sidebar history list */}
            <div className="lg:col-span-1">
              <History 
                history={history} 
                onClear={handleClearHistory} 
              />
            </div>
          </div>
        )}
      </main>

      {/* Minimalistic Footer */}
      <footer className="bg-white border-t border-slate-100 py-6 text-center text-xs text-slate-400">
        <div className="max-w-6xl mx-auto px-4 flex flex-col md:flex-row justify-between items-center gap-4">
          <p>&copy; {new Date().getFullYear()} DermaVision. Demonstrative evaluation software.</p>
          <div className="flex items-center gap-4">
            <span className="flex items-center gap-1">
              <ShieldAlert className="h-3.5 w-3.5" /> No PHI Retained
            </span>
            <span className="flex items-center gap-1">
              <UploadCloud className="h-3.5 w-3.5" /> Localhost Active
            </span>
          </div>
        </div>
      </footer>
    </div>
  );
}
