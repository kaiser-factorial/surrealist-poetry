# Windows PowerShell — CLI Reference

## Python

| Command | What it does |
|--------|--------------|
| `python --version` | Check if Python is installed and see the version |
| `python script.py` | Run a Python script |

---

## Pip (Python Package Installer)

| Command | What it does |
|--------|--------------|
| `pip install package_name` | Install a Python package |
| `pip install torch transformers matplotlib` | Install multiple packages at once |
| `pip install package_name -q` | Install quietly (less output) |

---

## Installing Software (winget)

| Command | What it does |
|--------|--------------|
| `winget --version` | Check if winget is available and see the version |
| `winget source update` | Refresh winget's package sources (run if packages aren't found) |
| `winget install Python.Python.3` | Install Python via winget |
| `winget install Microsoft.VisualStudioCode` | Install VS Code via winget |

> **Note:** If winget doesn't work, use the Microsoft Store or download directly from python.org / code.visualstudio.com

---

## Files & Folders

| Command | What it does |
|--------|--------------|
| `ls` | List files and folders in the current directory |
| `New-Item -ItemType Directory -Path "FolderName"` | Create a new folder |
| `New-Item -ItemType Directory -Path "Parent\Child" -Force` | Create nested folders in one command |
| `New-Item -ItemType File -Path "file.py"` | Create a new empty file |

---

## System Info

| Command | What it does |
|--------|--------------|
| `echo $env:PROCESSOR_ARCHITECTURE` | Check if CPU is x64, ARM64, or x86 |
| `winget --version` | Check winget version |
| `Get-ComputerInfo \| Select-Object CsName, OsArchitecture, CsNumberOfLogicalProcessors, CsTotalPhysicalMemory, OsVersion` | Get full system overview (CPU, RAM, OS) |
| `Get-CimInstance Win32_Processor \| Select-Object Name, LoadPercentage` | Check CPU model and current usage |
| `Get-CimInstance Win32_OperatingSystem \| Select-Object FreePhysicalMemory, TotalVisibleMemorySize` | Check RAM usage |
| `Get-CimInstance Win32_VideoController \| Select-Object Name, AdapterRAM, DriverVersion` | Check GPU model and driver |
| `nvidia-smi` | Check NVIDIA GPU usage in real time (only works if NVIDIA GPU is present) |
| `python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"` | Verify PyTorch can see and use the GPU |
| `taskmgr` | Open Task Manager (live CPU, GPU, RAM usage) |

---

## Microsoft Store (alternative to winget)

| Command | What it does |
|--------|--------------|
| `start ms-windows-store://search/?query=python` | Open Microsoft Store and search for Python |

---

## Packages Used in This Project

| Package | Install Command | What it does |
|--------|----------------|--------------|
| `transformers` | `pip install transformers` | Loads HuggingFace models including GPT-2 |
| `torch` (CPU only) | `pip install torch` | PyTorch — runs model training on CPU only |
| `torch` (GPU/CUDA) | `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124` | PyTorch with CUDA support — runs training on NVIDIA GPU (much faster) |
| `matplotlib` | `pip install matplotlib` | Plots graphs (entropy over time) |
| `beautifulsoup4` | `pip install beautifulsoup4` | Strips HTML tags from downloaded files |

---

## Notes
- On Windows, use `python` and `pip` (not `python3` or `pip3`)
- Run PowerShell as Administrator if you hit permission errors during installs
- If `winget` can't find a package, try `winget source update` first, then retry
