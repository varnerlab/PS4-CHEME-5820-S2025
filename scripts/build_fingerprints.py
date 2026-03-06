"""
build_fingerprints.py
---------------------
Reference script for generating 256-bit Morgan fingerprints for FDA-approved drugs
using RDKit (Python). This was used to generate the data/fda_drugs_fingerprints.jld2
file for CHEME 5820 PS4.

Requirements:
    pip install rdkit pandas h5py

Usage:
    python scripts/build_fingerprints.py \
        --smiles data/fda_drugs_smiles.csv \
        --output data/fda_drugs_fingerprints.h5

The output HDF5 file can be converted to JLD2 format using the companion
Julia script: scripts/convert_h5_to_jld2.jl

Morgan fingerprint parameters:
    - radius = 2  (Morgan/ECFP4 — encodes 2-hop chemical neighborhoods)
    - nBits = 256 (256-bit folded fingerprint)
    - useChirality = False
"""

import argparse
import numpy as np
import pandas as pd
import h5py
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

def smiles_to_morgan_fp(smiles: str, radius: int = 2, n_bits: int = 256) -> np.ndarray:
    """
    Convert a SMILES string to a binary Morgan fingerprint.

    Parameters
    ----------
    smiles : str
        SMILES string for the molecule.
    radius : int
        Morgan algorithm radius (2 = ECFP4).
    n_bits : int
        Length of the folded bit vector.

    Returns
    -------
    np.ndarray of shape (n_bits,), dtype float32
        Binary fingerprint (0 or 1 per bit).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits,
                                               useChirality=False)
    arr = np.zeros(n_bits, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def tanimoto(a: np.ndarray, b: np.ndarray) -> float:
    """
    Tanimoto (Jaccard) similarity between two binary fingerprints.
    J(a,b) = dot(a,b) / (||a||_1 + ||b||_1 - dot(a,b))
    """
    ab = np.dot(a, b)
    denom = a.sum() + b.sum() - ab
    return float(ab / denom) if denom > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description="Build Morgan fingerprints for FDA drugs")
    parser.add_argument("--smiles", default="data/fda_drugs_smiles.csv",
                        help="CSV with columns: name, drug_class, smiles")
    parser.add_argument("--output", default="data/fda_drugs_fingerprints.h5",
                        help="Output HDF5 file")
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--nbits", type=int, default=256)
    args = parser.parse_args()

    print(f"Loading SMILES from {args.smiles}...")
    df = pd.read_csv(args.smiles)
    print(f"  {len(df)} drugs loaded")

    print(f"Computing Morgan fingerprints (radius={args.radius}, nBits={args.nbits})...")
    fps = np.zeros((args.nbits, len(df)), dtype=np.float32)
    failed = 0
    for i, row in df.iterrows():
        fp = smiles_to_morgan_fp(row["smiles"], radius=args.radius, n_bits=args.nbits)
        fps[:, i] = fp
        if fp.sum() == 0:
            failed += 1

    print(f"  Failed to parse: {failed}/{len(df)}")
    print(f"  Mean bits set per drug: {fps.sum(axis=0).mean():.1f}")

    print(f"Saving to {args.output}...")
    with h5py.File(args.output, "w") as f:
        f.create_dataset("fingerprints", data=fps, compression="gzip")
        dt = h5py.special_dtype(vlen=str)
        f.create_dataset("drug_names", data=df["name"].values.astype(str))
        f.create_dataset("drug_classes", data=df["drug_class"].values.astype(str))
    print("Done.")


if __name__ == "__main__":
    main()
