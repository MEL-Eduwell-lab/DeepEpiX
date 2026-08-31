# 🗃️ Folder Structure

```
DeepEpiX/

├── data/                 # Put your data here - when built with Docker, local data directory is mounted on it.
│   ├── patient_1.ds      
│   ├── patient_2.fif
│   ├── patient_3_4D/
│   │   ├── rfDC_EEG
│   │   ├── config
│   │   └── hs_file

├── docs/                 # Use mkdocs

├── requirements/         # Use pip-tools to generate .txt from .in

├── src/                  
│   ├── assets/           # Static image/logo/icons
│   ├── cache-directory/  # Cached intermediate data or results - cleaned every time a new subect is loaded
│   ├── callbacks/        # Contains chainable functions that are automatically called whenever a UI element on viz.py page is changed
│   │   ├── utils/
│   │   │   ├── page1_utils.py 
│   │   │   ├── page2_utils.py 
│   │   │   ├── ...
│   │   │   └── pageN_utils.py
│   │   ├── page1_callbacks.py 
│   │   ├── page2_callbacks.py 
│   │   ├── ...
│   │   └── page3_callbacks.py  
│   ├── layout/           # Contains UI elements definition
│   │   ├── page1_layout.py 
│   │   ├── page2_layout.py 
│   │   ├── ...
│   │   └── pageN_layout.py  
│   ├── model_pipeline/   # Extracted from https://github.com/pmouches/DeepEpi/tree/main/pipeline with some modifications
│   ├── models/           # ML models, organized by modality
│   │   ├── MEG/           # models run on MEG recordings
│   │   └── EEG/           # models run on scalp EEG recordings
│   ├── pages/            # Multi-page app
│   │   ├── page1.py 
│   │   ├── page2.py 
│   │   ├── ...
│   │   └── pageN.py
│   ├── static/           # Static files
│   ├── config.py         # Configuration settings and constants
│   └── run.py            # Entry point to run the multi-page app


├── DeepEpiX.def          # Singularity definition file for containerization
├── Dockerfile            # Docker definition file for containerization
└── README.md 
```

This structure is schematic but aims to help you understand how the multi-page Dash app is organized.

Each page is defined in the `pages/` directory:

- The **layout** of each page is declared in `layout/`.
- The **interactivity (callbacks)** is handled in `callbacks/`.

> 🔎 Note: Since several pages share common callback functions, the callbacks, layout components, and utilities are organized by major components (e.g., `graph`, `history`, `ica`, `prediction`, `preprocessing`, etc.), rather than strictly by individual pages.

---