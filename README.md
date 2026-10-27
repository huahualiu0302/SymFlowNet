# SymFlowNet

Source code of the paper **Symmetry-Aware Deep Learning for Predicting Directional Pore-Scale Flow and Permeability in Digital Sandstone** (Yuxuan Xia, Chengqian Liu, Liang Jiao, Han Wang, Jianchao Cai).

SymFlowNet predicts the velocity fields of a two-dimensional rock slice in two flow directions from its pore image, with a single network trained on one lattice Boltzmann (LBM) simulation per training slice. It uses the transpose and reflection symmetry of creeping flow: the flow driven along y through a slice is the flow driven along x through the transposed slice, transposed back,

u_y(m) = [u_x(m^T)]^T.

A simulation in either direction therefore gives a training sample for one network that predicts the x-direction field, and the y-direction field is predicted from the transposed slice. The prediction is averaged over the four flipped versions of the slice.

## Repository structure

| Path | Content |
| --- | --- |
| `symflownet/models.py` | Encoder-decoder network with CBAM in the decoder, PatchGAN discriminator of the CycleGAN reference models |
| `symflownet/symmetry.py` | Transpose and flip transforms, prediction averaged over the flips |
| `symflownet/data.py` | Velocity scale and normalization, data split by depth intervals, data loading |
| `lbm/lbm_d2q9.py` | D2Q9 LBM solver on the GPU (NVIDIA Warp), TRT or BGK collision, lattice refinement |
| `lbm/run_labels.py` | Velocity labels of a stack of slices in the x- or y-direction |
| `lbm/channel_test.py` | Plane Poiseuille benchmark of the solver |
| `prepare_data.py` | Label files to training data, split and normalization constants |
| `train.py` | Training of SymFlowNet and of all reference models of the paper |
| `evaluate.py` | Pooled and per-slice R², relative error and permeability error on the test sets |
| `selftest.py` | Test on synthetic data, CPU only |

## Installation

Python 3.10 or later.

```
pip install -r requirements.txt
python selftest.py
```

`selftest.py` checks the symmetry relations, trains every model for a few iterations on synthetic data, checks that an interrupted run resumes to the same weights, evaluates the models and, if Warp is installed, runs the LBM solver on two small slices. It takes about 3 minutes on a CPU.

## Workflow

1. **Slices.** The micro-CT images are cut into non-overlapping 200 × 200 slices. For each rock, store the slices in `slices/<rock>.npz` with `solid` (N × 200 × 200, uint8, 1 in the grains) and `filenames`. The rocks are `BanderaGray`, `Berea`, `Castlegate`, `Parker` and `Bentheimer`. The 1000 Bandera Gray slices are 40 horizontal planes of 5 × 5 tiles, numbered so that slices 25l+1 to 25l+25 come from plane l. The depths of the planes are listed in `symflownet/data.py` and define the split into training, validation and test planes.

2. **LBM labels** (GPU). For every rock and direction:
   ```
   python lbm/run_labels.py --slices slices/Parker.npz --direction x --out labels
   python lbm/run_labels.py --slices slices/Parker.npz --direction y --out labels
   ```
   The default settings are those of the paper: TRT collision with magic parameter 3/16, 4 × 4 lattice nodes per pixel, kinematic viscosity 0.2 and pressure gradient 3.94 × 10⁻⁶ in lattice units of the original grid. A run that is stopped resumes from the finished slices.

3. **Training data.**
   ```
   python prepare_data.py --labels labels --out data
   ```

4. **Training.** For example SymFlowNet with three seeds:
   ```
   python train.py --data data --model symflownet --seed 0
   python train.py --data data --model symflownet --seed 1
   python train.py --data data --model symflownet --seed 2
   ```
   The models of the paper are `symflownet`, `separate`, `no_flip`, `no_cbam`, `symflownet_both`, `symflownet_half`, `symflownet_quarter`, `translation`, `translation_all`, `cyclegan` and `cyclegan_semi` (see the docstring of `train.py`). One SymFlowNet run of 25 000 iterations takes about 6 min on an NVIDIA GeForce RTX 5090.

5. **Evaluation.**
   ```
   python evaluate.py --data data --runs runs
   ```

## Data

The Bandera Gray, Parker, Castlegate and Bentheimer micro-CT datasets are available from the Digital Rocks Portal (https://www.digitalrocksportal.org). The Berea sandstone dataset is available from Imperial College London.

## License

MIT License, see `LICENSE`.
