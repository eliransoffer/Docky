#!/usr/bin/env python3
"""
Quick launcher for Streamlit RAG app
"""
import subprocess
import sys
import os

def main():
    """Launch Streamlit app"""
    
    # Check if .env file exists
    if not os.path.exists('.env'):
        print("⚠️  Warning: .env file not found!")
        print("Please create a .env file with your GOOGLE_API_KEY")
        print("Example: cp .env.example .env")
        print()
    
    # Check if streamlit is installed
    try:
        import streamlit
    except ImportError:
        print("❌ Streamlit not installed!")
        print("Install with: pip install -r requirements.txt")
        sys.exit(1)
    
    print("🚀 Starting RAG Streamlit App...")
    print("📱 The app will open in your browser automatically")
    print("🛑 Press Ctrl+C to stop the server")
    print()
    
    try:
        # Run streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.address", "localhost",
            "--server.port", "8501",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
    except Exception as e:
        print(f"❌ Error running Streamlit: {e}")

if __name__ == "__main__":
    main()