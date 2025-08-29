# I-JEPA Inversion Attacks

This project implements three black-box inversion techniques to reconstruct images from I-JEPA context embeddings using Stable Diffusion 2.1, as part of a research project. The techniques are Direct Optimization (DO), Decoder-Based (DB), and Diffusion Model-Based (DMB).

## Project Structure
```text
project_root/
├── models/
│   └── unet.py             # UNet model for inversion
├── scripts/
│   ├── train_do.py         # Training script for DO
│   ├── train_db.py         # Training script for DB
│   └── train_dmb.py        # Training script for DMB
├── attacks/
│   ├── attack_do.py        # Attack script for DO
│   ├── attack_db.py        # Attack script for DB
│   └── attack_dmb.py       # Attack script for DMB
├── data/
│   └── (pairs_do.npy, pairs_db.npy, pairs_dmb.npy)  # Generated image-embedding pairs
├── saved_models/
│   └── (do_inv.pth, db_inv.pth, dmb_inv.pth)        # Generated model weights
├── results/
│   └── (do_comparison.png, db_comparison.png, dmb_comparison.png)  # Generated comparison images
├── README.md
└── environment.yaml
```
## Dependencies

The project uses Conda to manage dependencies. The required packages are specified in `environment.yaml`.

### Key Dependencies
- PyTorch (`torch`, `torchvision`) for neural network operations
- Transformers (`transformers`) for I-JEPA model and processor
- Diffusers (`diffusers`) for Stable Diffusion 2.1 and VAE
- NumPy (`numpy`) for array handling
- Pillow (`pillow`) for image processing
- pytorch-msssim for SSIM metric
- lpips for LPIPS metric

## Setup

1. **Install Conda**:
Install Miniconda or Anaconda from [conda.io](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

2. **Create Conda Environment**:
   ```bash
   conda env create -f environment.yaml
   conda activate ijepa-inversion
   ```
3. **Set PYTHONPATH**:
  To resolve import issues (e.g., "No module named models"), ensure the project root is in your `PYTHONPATH`:
    ```bash
    export PYTHONPATH=$PYTHONPATH:$(pwd)  # On Windows: set PYTHONPATH=%PYTHONPATH%;%CD%
    ```

## Usage:
### Training:
Train the inversion models for each attack. Adjust --epochs, --batch_size, and --lr as needed.
```bash
python scripts/train_do.py --epochs 10 --batch_size 32 --lr 0.001
python scripts/train_db.py --epochs 10 --batch_size 8 --lr 0.001
python scripts/train_dmb.py --epochs 10 --batch_size 4 --lr 0.001
```
* Output:
  * Image-embedding pairs saved in `data/pairs_*.npy`
  * Trained model weights saved in `saved_models/*_inv.pth`.

### Running Attacks:
Run the attack scripts to reconstruct images and compute MSE loss. The scripts also save original vs. reconstructed image comparisons.
```bash
python attacks/attack_do.py
python attacks/attack_db.py
python attacks/attack_dmb.py
``` 
* Output:
  * Test MSE loss printed to console.
  * Comparison images saved in `results/*_comparison.png` (original images on the left, reconstructed on the right).