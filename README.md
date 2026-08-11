# 🏔 OSCAR-perma-ab

An abrupt permafrsot thaw emulator.

## 📚 Documentation

- [**Transitions between thaw states**](./docs/Transitions.png)

## 📝 Notes

This emulator is based on an inventory model described in [Turetsky et al. (2020)](https://www.nature.com/articles/s41561-019-0526-0)

## 🚀 Installation

Since this repository is not packaged for `pip`, users will need to clone the repository and run it directly within your Python environment.

### Prerequisites & Dependencies

This project was developed and tested on **Python 3.12** (requires **Python 3.11** or higher).

All required Python packages and their version constraints are listed in [`requirements.txt`](requirements.txt). 

Key dependencies include:

* **Data & Science:** `lmfit`, `numpy`, `pandas`, `scipy`, `xarray`
* **Visualization:** `matplotlib`, `seaborn`, `cartopy`

### Step-by-step
* **Create and Activate a Virtual Environment:**
It is strongly recommended to use an isolated environment to avoid package conflicts.

1. **Using `venv` (Standard Python):**
   ```bash
   # Create the environment
   python3 -m venv env

   # Activate the environment (Linux/macOS):
   source env/bin/activate
   ```

2. **Clone the repository:**
   ```bash
   git clone https://github.com/Xinrui-Rea/OSCAR-perma-ab.git
   cd OSCAR-perma-ab
   ```

3. **Install Required Packages:**
   ```Bash
   pip install -r requirements.txt
   ```
