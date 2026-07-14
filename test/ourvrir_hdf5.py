import sys
import h5py
import numpy as np
from pathlib import Path


def explorer_hdf5(chemin: Path) -> None:
    if not chemin.exists():
        print(f"[Erreur] Fichier introuvable : {chemin}")
        return

    print(f"\n{'='*60}")
    print(f" {chemin.name}")
    print(f"{'='*60}")

    with h5py.File(chemin, 'r') as f:

        def afficher(nom, objet):
            profondeur = nom.count('/')
            indent = "    " * profondeur
            nom_court = nom.split('/')[-1]

            if isinstance(objet, h5py.Group):
                print(f"{indent}📁 {nom_court}/  ({len(objet)} éléments)")

            elif isinstance(objet, h5py.Dataset):
                infos = f"shape={objet.shape}  dtype={objet.dtype}"

                # Aperçu des valeurs si petit dataset
                if objet.size <= 10:
                    valeurs = objet[:]
                    infos += f"  valeurs={valeurs}"
                else:
                    data = objet[:]
                    infos += f"  min={data.min():.4f}  max={data.max():.4f}  mean={data.mean():.4f}"

                # Attributs éventuels
                if objet.attrs:
                    attrs = {k: v for k, v in objet.attrs.items()}
                    infos += f"  attrs={attrs}"

                print(f"{indent} {nom_court}  |  {infos}")

        f.visititems(afficher)

    print(f"\n{'='*60}\n")


if __name__ == '__main__':

    chemin = Path("../data/brain_map_subj/sub-02_space-T1w_desc-ROImasks_voxelAnnotations.h5")
    explorer_hdf5(chemin)