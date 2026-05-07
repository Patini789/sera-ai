import os
import sys
import subprocess
import shutil
from pathlib import Path

def main():
    print("🌟 Starting Sera AI Quick Setup 🌟\n")
    
    base_dir = Path(__file__).parent.absolute()
    env_dir = base_dir / "env"
    
    # 1. Create Virtual Environment
    if not env_dir.exists():
        print("▶️ Creating virtual environment 'env'...")
        try:
            subprocess.run([sys.executable, "-m", "venv", "env"], check=True)
            print("✅ Virtual environment created successfully.\n")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to create virtual environment: {e}")
            sys.exit(1)
    else:
        print("✅ Virtual environment 'env' already exists.\n")
        
    # Determine pip path based on OS
    if os.name == 'nt': # Windows
        pip_path = env_dir / "Scripts" / "pip.exe"
    else: # Linux/Mac
        pip_path = env_dir / "bin" / "pip"
        
    # 2. Install Dependencies
    req_file = base_dir / "requirements.txt"
    if req_file.exists():
        print("▶️ Installing dependencies from requirements.txt...")
        try:
            subprocess.run([str(pip_path), "install", "-r", str(req_file)], check=True)
            print("✅ Dependencies installed successfully.\n")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            sys.exit(1)
    else:
        print("⚠️ requirements.txt not found. Skipping dependency installation.\n")

    # 3. Setup Configuration Files
    config_dir = base_dir / "Python" / "env"
    if config_dir.exists():
        print("▶️ Setting up configuration files...")
        
        # Special mappings if needed, otherwise just remove 'user.' prefix
        # user..env.txt -> .env
        
        for item in config_dir.iterdir():
            if item.is_file() and item.name.startswith("user."):
                # Handle special case for .env.txt
                if item.name == "user..env.txt":
                    target_name = ".env"
                else:
                    # Remove 'user.' prefix
                    target_name = item.name.replace("user.", "", 1)
                
                target_path = config_dir / target_name
                
                if not target_path.exists():
                    try:
                        shutil.copy2(item, target_path)
                        print(f"  [+] Created: {target_name} (from {item.name})")
                    except Exception as e:
                        print(f"  [x] Error copying {item.name}: {e}")
                else:
                    print(f"  [-] Skipped: {target_name} already exists.")
                    
        print("\n✅ Configuration files are ready.\n")
    else:
        print(f"⚠️ Configuration directory {config_dir} not found.\n")

    print("🎉 Setup Complete! 🎉")
    print("You can now run Sera AI by executing:")
    if os.name == 'nt':
        print("  env\\Scripts\\activate")
    else:
        print("  source env/bin/activate")
    print("  python -m Python\n")

if __name__ == "__main__":
    main()
