import subprocess, sys
# Install key packages into the venv
for pkg in ["numpy", "matplotlib", "scikit-learn"]:
    subprocess.run([sys.executable, "-m", "ensurepip", "--upgrade"], capture_output=True)
    break
# Show what's available
result = subprocess.run([sys.executable, "-c", "import sys; print(sys.executable); import importlib; mods=['numpy','matplotlib']; [print(m, importlib.util.find_spec(m)) for m in mods]"], capture_output=True, text=True)
print(result.stdout)
print(result.stderr)
