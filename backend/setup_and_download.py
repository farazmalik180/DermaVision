import os
import sys
import json
import subprocess

TOKEN = "KGAT_78dcdc99cc7a825e3eadeaf98eb2bf7c"

def setup_kaggle():
    print("--- Kaggle API Setup ---")
    # Read username from arguments or env variable, or ask user via stdin
    username = os.environ.get("KAGGLE_USERNAME")
    if not username:
        if len(sys.argv) > 1:
            username = sys.argv[1].strip()
        else:
            username = input("Please enter your Kaggle username: ").strip()
            
    if not username:
        print("Error: Kaggle username is required. Run script as: python setup_and_download.py <your_username>")
        sys.exit(1)
        
    # Determine ~/.kaggle/ path
    home = os.path.expanduser("~")
    kaggle_dir = os.path.join(home, ".kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)
    
    kaggle_json_path = os.path.join(kaggle_dir, "kaggle.json")
    
    creds = {
        "username": username,
        "key": TOKEN
    }
    
    # Write credentials
    with open(kaggle_json_path, "w") as f:
        json.dump(creds, f)
    os.environ["KAGGLE_API_TOKEN"] = TOKEN
        
    if os.name != 'nt':
        try:
            os.chmod(kaggle_json_path, 0o600)
        except Exception as e:
            print(f"Warning: could not set file permissions on kaggle.json: {e}")
            
    print(f"Successfully wrote Kaggle credentials to {kaggle_json_path}")
    
    # Install kaggle package if missing
    try:
        import kaggle
    except ImportError:
        print("Installing kaggle CLI...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])
        
    print("\nStarting dataset download (approx. 33GB)... This can take a while.")
    
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Run the download command using native Python API
    try:
        # Import after writing kaggle.json so it finds the new credentials
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        print("Kaggle API authenticated successfully.")
        
        print("Downloading dataset files...")
        api.dataset_download_files('kmader/skin-cancer-mnist-ham10000', path=data_dir, quiet=False)
        print("Download completed successfully!")
        
        zip_path = os.path.join(data_dir, "skin-cancer-mnist-ham10000.zip")
        if os.path.exists(zip_path):
            print("Extracting dataset...")
            import zipfile
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            print("Extraction complete!")
            # Remove zip to save disk space
            os.remove(zip_path)
            print("Deleted zip file to save disk space.")
    except Exception as e:
        print(f"Error during dataset download or extraction: {e}")
        print("Please verify that:")
        print("1. Your Kaggle username is correct.")
        print("2. You have access to the dataset at: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000")

if __name__ == "__main__":
    setup_kaggle()
