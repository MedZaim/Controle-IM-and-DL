# Étape 2 — README

This README explains how to run predictions using the trained models from Étape 1, both via the notebook `partie2.ipynb` and a simple CLI script. It also summarizes the project constraints for submission.

## Project Structure

- `../etape1/models/` — contains saved model weights:
  - `CustomCNN.pth`
  - `VGG16.pth`
  - `ViT-1.pth`
  - `ViT-2.pth`
- `etape2/partie2.ipynb` — notebook to load models and run predictions on one or more images.
- `models.py` — model definitions (CustomCNN, VGG16, ViT) located at the project root.
- `etape2/test_images/` — place any test images here (optional, used by the notebook).

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- Pillow
- pandas (if needed elsewhere)

You can install the minimum runtime dependencies:

```
pip install torch torchvision pillow
```

If you also need pandas (e.g., for data exploration):

```
pip install pandas
```

## Reproducibility Notes

- No transfer learning is used; models are implemented from scratch in `models.py`.
- Preprocessing is fixed to 64×64 resize and normalization to mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5] (producing inputs in [-1, 1]).
- Predictions are deterministic given fixed weights.

## Running the Notebook (partie2.ipynb)

1. Open `etape2/partie2.ipynb`.
2. Ensure the four model weight files are present in `../etape1/models/` with names exactly:
   - `CustomCNN.pth`, `VGG16.pth`, `ViT-1.pth`, `ViT-2.pth`.
3. Ensure `models.py` is at the project root (same level as `etape1` and `etape2`). The notebook adds `..` to `sys.path` to import it.
4. Put a test image in `etape2/test_images/` or set `test_image_path` in Cell 5 to point to any image.
5. Run all cells. The notebook will:
   - Import models from `models.py`.
   - Load weights from `../etape1/models/`.
   - Predict with each model and print the binary label: `Saine` (0) or `Malade` (1), with confidence.

Common path pitfalls:
- If you run the notebook from another working directory, ensure the relative paths still point to `../etape1/models/` and `etape2/test_images/`. You can replace them with absolute paths if needed, e.g., `C:\Users\LENOVO i7\Desktop\M2\Contrôle – IM & DL\etape1\models\VGG16.pth`.

## Command-line Prediction (predict.py)

If you prefer a script, create `predict.py` (or use the provided one, if present) with consistent preprocessing and paths. Example usage:

```
python predict.py --image etape2/test_images/ISIC_0024306_0.jpg
```

Expected output format:

```
==> Analyzing: ISIC_0024306_0.jpg
--------------------------------------------------
Custom CNN      → Saine (confiance: 0.812)
VGG16           → Malade (confiance: 0.674)
ViT-1           → Saine (confiance: 0.593)
ViT-2           → Saine (confiance: 0.701)
--------------------------------------------------
```

Make sure the script uses the same weight paths as the notebook:
- `../etape1/models/CustomCNN.pth`
- `../etape1/models/VGG16.pth`
- `../etape1/models/ViT-1.pth`
- `../etape1/models/ViT-2.pth`

## Submission Checklist

Compress and send `Nom_Prénom_Master.zip` containing:
- `models/` folder with the four `.pth` weight files (same names as above).
- `predict.py` script that loads these models and performs prediction on a given image path.
- `README.md` (this file) explaining setup and execution.

## Troubleshooting

- ModuleNotFoundError: `No module named 'models'`
  - Ensure `models.py` is at the project root and the notebook/script adds `..` or the root path to `sys.path`.
- FileNotFoundError for model weights:
  - Verify the exact filenames in `../etape1/models/` and adjust paths to match (`ViT-1.pth`, `ViT-2.pth`, not underscores).
- Image not found:
  - Provide a valid path to an image (absolute or relative to where you run the notebook/script).

## Notes

- Binary labels: 0 → Saine, 1 → Malade.
- The architectures and preprocessing must match between training (Étape 1) and inference (Étape 2).

