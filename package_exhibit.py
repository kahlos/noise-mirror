import os
import subprocess
import sys
import shutil
import urllib.request
import tarfile

# Configuration
DIST_DIR = "Deployment_v1"
PYTHON_URL = "https://github.com/indygreg/python-build-standalone/releases/download/20240713/cpython-3.12.4%2B20240713-aarch64-apple-darwin-install_only.tar.gz"
PYTHON_RUNTIME = os.path.join(DIST_DIR, "python_runtime")
LIBS_DIR = os.path.join(DIST_DIR, "libs")
APP_DIR = os.path.join(DIST_DIR, "app")
MODELS_DIR = os.path.join(DIST_DIR, "models")

def run(cmd, shell=True):
    print(f"Executing: {cmd}")
    subprocess.check_call(cmd, shell=shell)

def build():
    # 1. Clean & Create Structure
    if os.path.exists(DIST_DIR):
        shutil.rmtree(DIST_DIR)
    
    os.makedirs(LIBS_DIR)
    os.makedirs(APP_DIR)
    os.makedirs(MODELS_DIR)
    os.makedirs(PYTHON_RUNTIME)

    # 2. Download & Extract Portable Python
    print(f"Downloading portable Python from {PYTHON_URL}...")
    tar_path = "python_standalone.tar.gz"
    urllib.request.urlretrieve(PYTHON_URL, tar_path)
    
    print("Extracting Python runtime...")
    with tarfile.open(tar_path, "r:gz") as tar:
        # Extract into a temp folder and then move 'python' contents
        tar.extractall(path=DIST_DIR)
    
    # The tar usually contains a 'python' folder
    extracted_python_dir = os.path.join(DIST_DIR, "python")
    if os.path.exists(extracted_python_dir):
        # Move contents of Deployment_v1/python/ to Deployment_v1/python_runtime/
        for item in os.listdir(extracted_python_dir):
            shutil.move(os.path.join(extracted_python_dir, item), os.path.join(PYTHON_RUNTIME, item))
        os.rmdir(extracted_python_dir)
    
    os.remove(tar_path)

    # 3. Install Dependencies into /libs
    print("Installing dependencies into /libs...")
    python_bin = os.path.join(PYTHON_RUNTIME, "bin", "python3")
    
    # Guardrail 1: Upgrade pip, setuptools, and wheel
    print("Pillar 2 Guardrail: Upgrading pip, setuptools, wheel...")
    run(f"{python_bin} -m pip install --upgrade pip setuptools wheel")

    run(f"{python_bin} -m pip install -r requirements.txt --target {LIBS_DIR} --no-cache-dir")

    # 4. Copy Project Assets
    print("Copying project assets...")
    # Copy app files
    for item in os.listdir("app"):
        s = os.path.join("app", item)
        d = os.path.join(APP_DIR, item)
        if os.path.isdir(s):
            shutil.copytree(s, d)
        else:
            shutil.copy2(s, d)
    
    # Copy local models cache if it exists to fast-track packaging
    if os.path.exists("models"):
        print("Copying local models cache...")
        shutil.copytree("models", MODELS_DIR, dirs_exist_ok=True, symlinks=True)

    # Pre-download models to guarantee offline support
    print("Pre-downloading AI Models for offline usage in target bundle...")
    download_script = f"""
import sys
import os
sys.path.insert(0, '{os.path.abspath(LIBS_DIR)}')
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("Failed to import huggingface_hub. Aborting model download.")
    sys.exit(1)

models_dir = '{os.path.abspath(MODELS_DIR)}'
print("Downloading SD 1.5...")
snapshot_download(repo_id='runwayml/stable-diffusion-v1-5', cache_dir=models_dir)
print("Downloading ControlNet Canny...")
snapshot_download(repo_id='lllyasviel/sd-controlnet-canny', cache_dir=models_dir)
print("Downloading LCM LoRA...")
snapshot_download(repo_id='latent-consistency/lcm-lora-sdv1-5', cache_dir=models_dir)
print("Offline model cache fully populated.")
"""
    with open("dl_models.py", "w") as f:
        f.write(download_script)
    run(f"{python_bin} dl_models.py", shell=True)
    os.remove("dl_models.py")

    # 5. Create Launcher Script
    print("Creating Launch_Exhibit.command...")
    launcher_content = f"""#!/bin/bash
# 1. Dynamically resolve the directory
DIR="$( cd "$( dirname "${{BASH_SOURCE[0]}}" )" && pwd )"

# 2. Strip macOS Gatekeeper quarantine recursively and silently
xattr -rc "$DIR" 2>/dev/null

# 3. Ensure the portable Python is executable
chmod -R u+x "$DIR/python_runtime/bin" 2>/dev/null

# 4. Force Python to use our isolated dependencies
export PYTHONPATH="$DIR/libs"

# 5. Launch
"$DIR/python_runtime/bin/python3" "$DIR/app/main_exhibit.py"
"""
    launcher_path = os.path.join(DIST_DIR, "Launch_Exhibit.command")
    with open(launcher_path, "w") as f:
        f.write(launcher_content)
    
    os.chmod(launcher_path, 0o755)

    print("\n--- Packaging Complete ---")
    print(f"Output Directory: {DIST_DIR}")

if __name__ == "__main__":
    build()
