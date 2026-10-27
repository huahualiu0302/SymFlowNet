"""Assemble the label files into the training data, data split and normalization constants.

    python prepare_data.py --labels labels --out data

Expects <labels>/<rock>_x.npz and <labels>/<rock>_y.npz for BanderaGray, Berea, Castlegate, Parker and Bentheimer
(see symflownet/data.py and lbm/run_labels.py). Writes <out>/data.npz, split.json and norm.json.
"""
import argparse

from symflownet.data import build_dataset

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--labels', default='labels')
    ap.add_argument('--out', default='data')
    a = ap.parse_args()
    build_dataset(a.labels, a.out)
