# Face Classifier — Ghaith / Aziz / Unidentified

A PyTorch image classifier that fine-tunes a pretrained ResNet18 to recognize photos of two specific people (Ghaith, Aziz) and flag everyone else as "unidentified."

## Overview

The model uses transfer learning: a ResNet18 backbone pretrained on ImageNet has its final layer replaced with a 3-class classifier and is fine-tuned on a small custom dataset. This approach works well even with a modest number of training photos per class.

**Classes:** `ghaith`, `aziz`, `unidentified`

## Project structure

```
.
├── data/
│   ├── train/
│   │   ├── ghaith/
│   │   ├── aziz/
│   │   └── unidentified/
│   └── val/
│       ├── ghaith/
│       ├── aziz/
│       └── unidentified/
├── train.py
├── test.py
├── requirements.txt
└── README.md
```

`data/` is not tracked in this repo (see `.gitignore`) since it contains personal photos.

## Setup

This project was built and tested with Python 3.14.

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# swap the URL above for a CUDA index if you have an NVIDIA GPU
pip install -r requirements.txt
```

## Preparing the dataset

Create the folder structure above and populate each class folder with photos:

- **ghaith / aziz**: 60-100+ photos each, mostly front-facing with some angle variation, varied lighting/backgrounds/expressions. Keep the two roughly balanced in count.
- **unidentified**: 100-200+ photos of as many different people as possible — diversity of identity matters more than pose variety here, since this class has to generalize to anyone not in the other two classes.

Crop images to focus on the face where possible for better accuracy.

## Training

```bash
python train.py --data_dir data --epochs 15 --batch_size 16
```

Key options:
- `--unfreeze` — fine-tune the whole backbone instead of just the final layer (use only with 150+ images/class)
- `--out model_name.pt` — set a custom output filename (default: `face_classifier.pt`)
- `--lr` — learning rate (default: `1e-3`)

The script saves the best-performing checkpoint (by validation accuracy) along with the class-index mapping, so `test.py` doesn't need to guess it.

Re-running this command retrains from scratch using whatever is currently in `data/` — it does not incrementally update an existing checkpoint.

## Testing / inference

```bash
python test.py --model face_classifier.pt --image path/to/photo.jpg
python test.py --model face_classifier.pt --folder path/to/folder --threshold 0.6
```

`--threshold` sets a minimum confidence for a Ghaith/Aziz prediction to be trusted; below it, the result falls back to "unidentified (low confidence)" — useful since real-world faces outside the training set will sometimes get a confident-looking wrong answer otherwise.

## Notes & limitations

- This is an open-set problem in practice (anyone in the world could appear in a test photo), so the "unidentified" class can only ever approximate "not Ghaith, not Aziz" based on the diversity it saw during training.
- Model accuracy depends heavily on matching the training photo distribution (camera type, lighting, framing) to how it'll actually be used.

## Credits

Ghaith Hajji, IT and Data Analytics Senior Student at Tunis Business School.
