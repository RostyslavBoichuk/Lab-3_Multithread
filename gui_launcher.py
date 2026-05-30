##\file gui_launcher.py
##\brief GUI application launcher
##\details Entry point for running the application in GUI mode
##\author Lab Team
##\version 1.0

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.gui import run_gui

if __name__ == '__main__':
    ##\brief Main entry point for GUI application
    print("Starting GUI Application...")
    run_gui()
