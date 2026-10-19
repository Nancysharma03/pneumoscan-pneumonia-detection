# PneumoScan — Streamlit UI

A Streamlit front end for the chest X-ray pneumonia classification project. It loads your **existing trained CNN checkpoint** and predicts the notebook's two labels: `NORMAL` and `PNEUMONIA`.

## 1. Project structure

```text
pneumonia_streamlit_app/
├── app.py
├── requirements.txt
└── models/
    └── best_cnn_model.keras
```

Copy your trained `best_cnn_model.keras` file into the `models` folder. The model file is not included in this starter package because it must come from your training environment.

## 2. Install and run (Windows PowerShell)

Open the terminal in this folder, then run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation, you can run the environment's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`.

## 3. Save/download the model from Colab

After training, save the CNN model to Google Drive:

```python
cnn_model.save("/content/drive/MyDrive/Pneumonia_Project/models/best_cnn_model.keras")
```

Download `best_cnn_model.keras` from Drive and put it in the local `models/` folder. If your checkpoint has a different location, set `PNEUMONIA_MODEL_PATH` to that path before starting Streamlit.

## Important implementation notes

- The notebook's directory generator maps `NORMAL` to `0` and `PNEUMONIA` to `1`.
- Training images were passed through RGB loading, resized to `224 × 224`, and rescaled by `1/255`; inference uses the same basic input preparation.
- The UI uses the CNN checkpoint. The notebook reported **90.87% test accuracy** for CNN and **72.60%** for ResNet50, so CNN is the default selected model for this demo.
- The notebook also includes grayscale/histogram-equalization examples, but its actual training generator uses RGB images with rescaling and augmentation. The app follows the generator input pipeline rather than applying a different enhancement pipeline.
- This is an educational prototype, not a medical device. Never use it to make clinical decisions.
