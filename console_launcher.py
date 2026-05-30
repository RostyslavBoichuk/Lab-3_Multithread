##\file console_launcher.py
##\brief Console application launcher
##\details Entry point for running the application in console mode
##\author Lab Team
##\version 1.0

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

if __name__ == '__main__':
    ##\brief Main entry point for console application
    print("Starting Console Application...")
    from main import run_console_demo
    run_console_demo()
