import sys
import os

# Prüfen, ob wir im 'Sandkasten' (.venv) sind
interpreter_path = sys.executable
is_venv = ".venv" in interpreter_path

print("--- System-Check ---")
print(f"Python läuft aus: {interpreter_path}")
print(f"Virtuelle Umgebung aktiv: {'JA ✅' if is_venv else 'NEIN ❌'}")

try:
    import numpy as np
    import scipy
    print("Mathematik-Bibliotheken: Bereit ✅")
except ImportError:
    print("Mathematik-Bibliotheken: Fehlen noch ❌")