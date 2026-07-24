"""
AeroVision CFD v1.0 - Crash Logger
Standalone error capture module for application stability.
"""

import sys
import traceback
from datetime import datetime
from pathlib import Path

def exception_handler(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    log_path = Path(__file__).parent / "cfd_error_report.log"
    
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n" + "="*60 + "\n")
        f.write(f"AEROCFD CRASH REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*60 + "\n")
        f.write(f"Error Type: {exc_type.__name__}\n")
        f.write(f"Error Message: {exc_value}\n")
        f.write("\n--- Traceback ---\n")
        traceback.print_tb(exc_traceback, file=f)
        f.write("="*60 + "\n")

    print(f"\nSystem error detected. Log saved to: {log_path.name}")
    sys.__excepthook__(exc_type, exc_value, exc_traceback)

def register_logger():
    sys.excepthook = exception_handler