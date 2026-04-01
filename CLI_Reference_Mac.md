# Mac Terminal — CLI Reference

## Python

| Command | What it does |
|--------|--------------|
| `python3 --version` | Check if Python is installed and see the version |
| `python3 script.py` | Run a Python script |
| `python3 -c "..."` | Run a short Python snippet directly in the terminal without a file |

---

## Pip (Python Package Installer)

| Command | What it does |
|--------|--------------|
| `pip3 install package_name` | Install a Python package |
| `pip3 install torch transformers matplotlib` | Install multiple packages at once |
| `pip3 install package_name -q` | Install quietly (less output) |

---

## Files & Folders

| Command | What it does |
|--------|--------------|
| `ls` | List files and folders in the current directory |
| `ls path/to/folder` | List contents of a specific folder |
| `cd path/to/folder` | Navigate into a folder |
| `mkdir folder_name` | Create a new folder |
| `mkdir -p parent/child` | Create nested folders in one command |

---

## Packages Used in This Project

| Package | Install Command | What it does |
|--------|----------------|--------------|
| `transformers` | `pip3 install transformers` | Loads HuggingFace models including GPT-2 |
| `torch` | `pip3 install torch` | PyTorch — runs model training |
| `matplotlib` | `pip3 install matplotlib` | Plots graphs (entropy over time) |
| `beautifulsoup4` | `pip3 install beautifulsoup4` | Strips HTML tags from downloaded files |

---

## Notes
- On Mac, always use `python3` and `pip3` (not `python` or `pip`) to ensure you're using the right version
- If a package install shows PATH warnings, the package still installed correctly
