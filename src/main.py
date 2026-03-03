import sys
import subprocess
from pathlib import Path

def run():
    """
    Main entry point: initialize databases and run the Streamlit app.
    """
    print("[INFO] Starting Streamlit application...")
    app_path = Path(__file__).parent / "app.py"
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)], check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to start Streamlit application: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run()