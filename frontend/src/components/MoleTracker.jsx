import React, { useState, useEffect } from 'react';
import { Plus, Trash2, Folder, Clock } from 'lucide-react';
import { getProfiles, createProfile, deleteProfile, getScansForProfile } from '../db';

export default function MoleTracker() {
  const [profiles, setProfiles] = useState([]);
  const [selectedProfile, setSelectedProfile] = useState(null);
  const [scans, setScans] = useState([]);
  const [newProfileName, setNewProfileName] = useState('');

  useEffect(() => {
    loadProfiles();
  }, []);

  const loadProfiles = async () => {
    const p = await getProfiles();
    setProfiles(p);
  };

  const handleCreateProfile = async (e) => {
    e.preventDefault();
    if (!newProfileName.trim()) return;
    await createProfile(newProfileName.trim());
    setNewProfileName('');
    loadProfiles();
  };

  const handleDeleteProfile = async (id, e) => {
    e.stopPropagation();
    if (window.confirm('Are you sure you want to delete this profile and all its history?')) {
      await deleteProfile(id);
      if (selectedProfile?.id === id) {
        setSelectedProfile(null);
      }
      loadProfiles();
    }
  };

  const handleSelectProfile = async (profile) => {
    setSelectedProfile(profile);
    const s = await getScansForProfile(profile.id);
    setScans(s);
  };

  return (
    <div className="bg-white rounded-2xl shadow-sm border border-slate-100 p-6">
      <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
        <Folder className="text-teal-600" />
        Mole Tracker Profiles
      </h2>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Profiles List */}
        <div className="md:col-span-1 border-r border-slate-100 md:pr-4">
          <form onSubmit={handleCreateProfile} className="flex gap-2 mb-4">
            <input 
              type="text" 
              placeholder="E.g., Left Arm Mole" 
              value={newProfileName}
              onChange={(e) => setNewProfileName(e.target.value)}
              className="flex-1 text-sm rounded-lg border border-slate-200 px-3 py-2 outline-none focus:border-teal-500"
            />
            <button type="submit" className="bg-teal-600 text-white p-2 rounded-lg hover:bg-teal-700 transition">
              <Plus className="h-5 w-5" />
            </button>
          </form>

          <div className="space-y-2 max-h-[500px] overflow-y-auto pr-1">
            {profiles.length === 0 ? (
              <p className="text-sm text-slate-500 text-center py-4">No profiles created yet.</p>
            ) : (
              profiles.map(p => (
                <div 
                  key={p.id}
                  onClick={() => handleSelectProfile(p)}
                  className={`p-3 rounded-xl border cursor-pointer flex justify-between items-center transition ${
                    selectedProfile?.id === p.id ? 'border-teal-500 bg-teal-50' : 'border-slate-100 hover:border-teal-300'
                  }`}
                >
                  <span className="font-semibold text-slate-800 truncate pr-2">{p.name}</span>
                  <button onClick={(e) => handleDeleteProfile(p.id, e)} className="text-slate-400 hover:text-red-500 shrink-0">
                    <Trash2 className="h-4 w-4" />
                  </button>
                </div>
              ))
            )}
          </div>
        </div>

        {/* Selected Profile Details */}
        <div className="md:col-span-2 md:pl-2">
          {selectedProfile ? (
            <div>
              <h3 className="text-xl font-bold mb-4">{selectedProfile.name} - History</h3>
              {scans.length === 0 ? (
                <p className="text-sm text-slate-500">No scans saved to this profile yet. Run a scan and click "Save to Profile".</p>
              ) : (
                <div className="space-y-4 max-h-[600px] overflow-y-auto pr-2">
                  {scans.map(scan => (
                    <div key={scan.id} className="border border-slate-100 rounded-xl p-4 flex flex-col md:flex-row gap-4 items-start hover:shadow-md transition bg-slate-50/50">
                      <img src={scan.imageBase64} alt="Scan" className="w-full md:w-32 h-32 object-cover rounded-lg shadow-sm" />
                      <div className="flex-1">
                        <div className="flex items-center gap-2 mb-2">
                          <Clock className="h-4 w-4 text-slate-400" />
                          <span className="text-xs text-slate-500 font-medium">{new Date(scan.date).toLocaleString()}</span>
                        </div>
                        <h4 className="font-bold text-lg text-slate-800">{scan.result.label}</h4>
                        
                        <div className="flex flex-wrap items-center gap-2 mt-3">
                          <span className={`text-xs font-bold px-2.5 py-1 rounded-full ${
                            scan.result.risk_level === 'High Risk' ? 'bg-red-100 text-red-700' :
                            scan.result.risk_level === 'Moderate Risk' ? 'bg-orange-100 text-orange-700' :
                            'bg-green-100 text-green-700'
                          }`}>
                            {scan.result.risk_level}
                          </span>
                          <span className="text-xs font-semibold text-slate-600 bg-white border border-slate-200 px-2.5 py-1 rounded-full">
                            Conf: {(scan.result.confidence * 100).toFixed(1)}%
                          </span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center h-full text-slate-400 py-12">
              <Folder className="h-12 w-12 mb-3 opacity-20" />
              <p>Select a profile to view its history</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
