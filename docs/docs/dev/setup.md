## 🛠 Local Installation (for Development Mode)

1. **Clone the Repository in your working directory:**
```bash
git clone https://github.com/mel-eduwell-lab/DeepEpiX.git
```

2. **Set up the Dash Environment:**
```bash
cd DeepEpiX
python3 -m venv .dashenv
source .dashenv/bin/activate
python3 -m pip install -r requirements/requirements-dashenv.txt
deactivate
```

3. **Set up Prediction Model Environments:**
```bash
python3 -m venv .tfenv
source .tfenv/bin/activate
# For Non Macos Users
python3 -m pip install -r requirements/requirements-tfenv-cuda.txt
# For Macos Users
python3 -m pip install -r requirements/requirements-tfenv-macos.txt
deactivate
```
```bash
python3 -m venv .torchenv
source .torchenv/bin/activate
# For Non Macos Users
python3 -m pip install -r requirements/requirements-torchenv-cuda.txt
# For Macos Users
python3 -m pip install -r requirements/requirements-torchenv-macos.txt
deactivate
```
4. **Activate your Dash Environment and Start Running the App:**
```bash
source .dashenv/bin/activate
python3 src/run.py
```
Then, open the app in your web browser at:
[http://localhost:8050/](http://localhost:8050/)

🥳 You can start editing code while visualizing automatic reloads (if DEBUG is set to True in `config.py`). 

> For quick access, ensure that your MEG/EEG data is placed in the data folder within the project directory.

## 🗒️ Development Notes

- 🐍 **Python Version**:  
  DeepEpiX was developed using **Python 3.9**. It is recommended to use this version to ensure compatibility.

- 🧰 **Recommended Developer Tools**:
    - [`pip-tools`](https://github.com/jazzband/pip-tools):  
        Helps manage and compile `requirements.in` into a clean, version-pinned `requirements.txt`.
        ```bash
        pip install pip-tools
        pip-compile --output-file=requirements.txt requirements.in
        ```

    - [`mkdocs`](https://www.mkdocs.org/):  
        Used to build clean, static documentation sites from Markdown files.
        ```bash
        pip install mkdocs
        mkdocs serve  # Preview the docs locally
        mkdocs build  # Generate the static site
        ```