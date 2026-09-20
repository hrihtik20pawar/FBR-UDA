"""
Quick setup script for FBR-UDA project.
Creates virtual environment and installs dependencies.

Usage:
  python setup.py
"""

import os
import sys
import subprocess
from pathlib import Path


def run_cmd(cmd, desc=""):
    """Run a shell command."""
    print(f"\n[setup] {desc}")
    print(f"  Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [error] {result.stderr}")
        return False
    print(f"  [done] {desc}")
    return True


def main():
    project_root = Path(__file__).parent
    
    print("="*60)
    print("FBR-UDA: Quick Setup")
    print("="*60)
    
    # 1. Create virtual environment
    venv_dir = project_root / ".venv"
    if not venv_dir.exists():
        print("\n[1/3] Creating virtual environment...")
        run_cmd(f'python -m venv "{venv_dir}"', "Creating .venv")
    else:
        print("\n[1/3] Virtual environment already exists.")
    
    # 2. Activate and install requirements
    if sys.platform == "win32":
        activate_script = venv_dir / "Scripts" / "activate.bat"
        pip_path = venv_dir / "Scripts" / "pip.exe"
        python_path = venv_dir / "Scripts" / "python.exe"
    else:
        activate_script = venv_dir / "bin" / "activate"
        pip_path = venv_dir / "bin" / "pip"
        python_path = venv_dir / "bin" / "python"
    
    print("\n[2/3] Installing dependencies...")
    run_cmd(f'"{pip_path}" install --upgrade pip', "Upgrading pip")
    run_cmd(f'"{pip_path}" install -r requirements.txt', "Installing requirements")
    
    # 3. Install segment-anything
    print("\n[3/3] Installing segment-anything...")
    run_cmd(f'"{pip_path}" install git+https://github.com/facebookresearch/segment-anything.git', "Installing SAM")
    
    print("\n" + "="*60)
    print("SETUP COMPLETE!")
    print("="*60)
    print(f"\nTo activate the environment:")
    if sys.platform == "win32":
        print(f"  {venv_dir}\\Scripts\\activate")
    else:
        print(f"  source {venv_dir}/bin/activate")
    
    print("\nNext steps:")
    print("1. Activate the environment")
    print("2. Run: python download_all.py")
    print("3. Run the FBR notebook")
    print("4. Run: python main.py")


if __name__ == "__main__":
    main()
