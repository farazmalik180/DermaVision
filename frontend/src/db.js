import localforage from 'localforage';

localforage.config({
  name: 'DermaVision',
  storeName: 'mole_tracker'
});

// Profiles: { id, name, createdAt }
export const getProfiles = async () => {
  const profiles = await localforage.getItem('profiles');
  return profiles || [];
};

export const createProfile = async (name) => {
  const profiles = await getProfiles();
  const newProfile = {
    id: Date.now().toString(),
    name,
    createdAt: new Date().toISOString()
  };
  profiles.push(newProfile);
  await localforage.setItem('profiles', profiles);
  return newProfile;
};

export const deleteProfile = async (id) => {
  let profiles = await getProfiles();
  profiles = profiles.filter(p => p.id !== id);
  await localforage.setItem('profiles', profiles);
  
  // Clean up associated scans
  let scans = await getScans();
  scans = scans.filter(s => s.profileId !== id);
  await localforage.setItem('scans', scans);
  
  return profiles;
};

// Scans: { id, profileId, imageBase64, result, date }
export const getScans = async () => {
  const scans = await localforage.getItem('scans');
  return scans || [];
};

export const getScansForProfile = async (profileId) => {
  const scans = await getScans();
  return scans.filter(s => s.profileId === profileId).sort((a, b) => new Date(b.date) - new Date(a.date));
};

export const addScan = async (profileId, imageBase64, result) => {
  const scans = await getScans();
  const newScan = {
    id: Date.now().toString(),
    profileId,
    imageBase64,
    result,
    date: new Date().toISOString()
  };
  scans.push(newScan);
  await localforage.setItem('scans', scans);
  return newScan;
};
