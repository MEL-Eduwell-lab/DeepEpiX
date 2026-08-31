# Tuto 6: Add your own model

This tutorial explains how to integrate a **custom model** into the Dash app.  
Models are executed in **separate environments** to maintain a lightweight footprint and prevent dependency conflicts

The primary advantage of the DeepEpiX architecture is the clean separation between the user interface and the inference logic. Researchers can integrate new deep learning models solely by updating the backend pipeline scripts, requiring **no knowledge** of Dash, HTML, or CSS.

To do so, you need to install the app locally because it requires code adjustments to make your custom model pipeline work. 
[↪️ Go to Developer Guide > Setup & Run](../dev/setup.md)

---

## Folder Structure

- **Dash app environment**:  
  `.dashenv` → Runs the Dash UI and callbacks.  

- **Model environments**:  
    `.tfenv` → TensorFlow models (`.keras`, `.h5`)  
    `.torchenv` → PyTorch models (`.pth`, `.ckpt`)  

- **Model folder**:  
  Place your model inside `models/<MODALITY>/` (`models/MEG/` or `models/EEG/`)
  with the correct extension. The subfolder tells the app which modality the
  model applies to, so it is only offered for matching recordings.

---

## How It Works

When you run a model:

1. The app detects the model type based on the file extension and modality folder.
2. It selects the corresponding Python binary (`.tfenv` or `.torchenv`).
3. A subprocess is started to execute the model script (`model_pipeline/main.py`).
4. Results are saved to the cache directory for later retrieval.

---

## 🚀 Adding Your Model

Follow these steps to integrate your trained model into the inference pipeline

**1. Add Your Model File**

Place your trained model inside the matching modality subfolder,
`models/MEG/` or `models/EEG/`:

- TensorFlow → `.keras` or `.h5`  
- PyTorch → `.pth` or `.ckpt`

**2. Configure Environments (if needed)**

Update `config.py` to point to your virtual environments:
```python
TENSORFLOW_ENV = Path("/path/to/.tfenv")
TORCH_ENV = Path("/path/to/.torchenv")
```

- If you followed the setup tutorial, no changes are required.
- Install any additional dependencies your model needs inside the corresponding environment.

**3. Create an Inference Pipeline**

Add a model-specific inference script in the `model_pipeline/` directory:

```bash
model_pipeline/run_<model_name>.py
```
This script should handle:

- Input preprocessing
- Model loading
- Inference logic
- Output formatting
- Prediction saving in the cache directory for later retrieval

You can also reuse or adapt an existing pipeline. See the example:

```bash
model_pipeline/run_example.py
```

**4. Register Your Model**

In `model_pipeline/main.py`, map your model name to its pipeline by updating `MODEL_MODULES`:

```python
MODEL_MODULES = {
    "your/model/name": "model_pipeline.run_model_name",
}
```

**5. Add Custom Logic (Optional)**

If your model requires a custom workflow or additional parameters, extend the `run_model_pipeline` function:

```python
if model_name == "your/model/name":
    return test_model_custom(...)
```

Use this only when the default pipeline is not sufficient.

**6. Match Training Configuration**

To ensure consistent results between training and inference, define the same filtering parameters used during training in:

```bash
static/model_config.json
```

This will be use when making inference using the "same as training" configuration.

The file is keyed by modality then file name, so a checkpoint with the same
name in `models/MEG/` and `models/EEG/` can carry different filters:

```json
{
  "EEG": {
    "your_model.ckpt": { "sfreq": 200, "highpass": 10.0, "lowpass": 70.0, "notch": 50.0 }
  }
}
```

Supported keys: `sfreq`, `highpass`, `lowpass`, `notch`. The referential /
bipolar choice is *not* set here: it is defined in the model pipeline's
`hparams` file and applied by the model itself from the referential signal.

---

## ✅ Checklist Before Running

* Add your model to `models/MEG/` or `models/EEG/`.
* (Optional) Configure environments in `config.py`.
* Create a pipeline script in `model_pipeline/`.
* Register your model in `MODEL_MODULES`
* Extend the pipeline if needed
* Define training configuration in `model_config.json`


---