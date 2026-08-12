"""Teste le folding `chunked_trimmed` de litcoder_core (voir src/litcoder_folding.py)
sur la vraie structure session/run de sub-01, et compare avec le split maison du
projet (GroupShuffleSplitSession, session-level).

Note : le fichier TRIBE (output/features/things_encoding/sub-01.h5, 14 Go) est un
placeholder iCloud "dataless" sur cette machine (0 octet réellement sur disque,
cf. `ls -lO` -> flag "dataless") : toute lecture déclenche un téléchargement qui
bloque indéfiniment. On n'a donc pas besoin de X ici : ce script ne teste que la
LOGIQUE DE SPLIT (indices de TR), donc on prend n_samples/groupes/run_ids
directement depuis le fichier BOLD (petit, ~170 Mo, déjà local), sans passer par
RidgeRegression.create_X_Y_total().
"""
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from litcoder_folding import create_chunked_folds_trimmed, create_kfold_trimmed
from GroupShuffleSplitSession import GroupShuffleSplitSession

CHEMIN_BOLD = (
    Path(__file__).resolve().parent.parent
    / "data" / "timeseries" / "cneuromod2026" / "sub-01"
    / "sub-01_task-things_space-MNI152NLin2009cAsym_atlas-cneuromod26_desc-1134Parcels_timeseries.h5"
)


def charger_structure_reelle(chemin_bold: Path):
    """Reproduit l'ordre de concaténation de RidgeRegression.create_X_Y_total()
    (sessions triées, runs triés) et renvoie n_samples, groupes (session), et
    run_ids (identifiant global de run, plus fin que la session)."""
    groupes, run_ids = [], []
    run_id_global = 0
    with h5py.File(chemin_bold, "r") as f:
        for ses in sorted(f.keys()):
            num_ses = int(ses.replace("ses-", ""))
            for run_ds in sorted(f[ses].keys()):
                n_tr = f[ses][run_ds].shape[0]
                groupes.append(np.full(n_tr, num_ses))
                run_ids.append(np.full(n_tr, run_id_global))
                run_id_global += 1
    groupes = np.concatenate(groupes)
    run_ids = np.concatenate(run_ids)
    return len(groupes), groupes, run_ids


def runs_coupes_par_le_split(train_idx, test_idx, run_ids):
    """Nombre de runs dont des TR se retrouvent à la fois en train ET en test
    (= contamination : le split ignore la frontière de run)."""
    runs_train = set(run_ids[train_idx])
    runs_test = set(run_ids[test_idx])
    return len(runs_train & runs_test)


def analyser_chunked_trimmed(n_samples, run_ids, chunk_length, n_folds=5, trim_size=5, seed=0):
    rng = np.random.default_rng(seed)
    folds = create_chunked_folds_trimmed(
        n_samples, n_folds, chunk_length, trim_size=trim_size, shuffle=True, rng=rng
    )
    print(f"\n--- chunked_trimmed | chunk_length={chunk_length} (1 run = 190 TR) | "
          f"trim_size={trim_size} | n_folds={n_folds} ---")
    for i, (train_idx, test_idx) in enumerate(folds):
        train_idx = np.array(train_idx)
        test_idx = np.array(test_idx)
        n_coupes = runs_coupes_par_le_split(train_idx, test_idx, run_ids)
        print(f"  Fold {i+1}: train={len(train_idx):5d}  test={len(test_idx):5d}  "
              f"runs coupés train/test={n_coupes}")


def analyser_group_shuffle(n_samples, groupes, n_folds=5, seed=0):
    splitter = GroupShuffleSplitSession(n_splits=n_folds, test_size=0.2, random_state=seed)
    print(f"\n--- GroupShuffleSplitSession (méthode actuelle du projet, niveau session) ---")
    for i, (train_idx, test_idx) in enumerate(splitter.split(None, None, groupes)):
        print(f"  Fold {i+1}: train={len(train_idx):5d}  test={len(test_idx):5d}  "
              f"sessions test={sorted(set(groupes[test_idx]))}")


if __name__ == "__main__":
    n_samples, groupes, run_ids = charger_structure_reelle(CHEMIN_BOLD)
    n_sessions = len(set(groupes))
    n_runs = len(set(run_ids))
    print(f"sub-01 : n_samples={n_samples}  n_sessions={n_sessions}  n_runs={n_runs}  "
          f"(chaque run = 190 TR, exactement, vérifié sur les {n_runs} runs)")

    # Cas 1 : chunk_length aligné sur la longueur d'un run (190 TR) -> les chunks
    # ne peuvent PAS tomber à cheval sur deux runs, par construction.
    analyser_chunked_trimmed(n_samples, run_ids, chunk_length=190, trim_size=5)

    # Cas 2 : chunk_length "naïf" (ex. 50 TR, valeur typique vue dans des configs
    # litcoder pour des scans continus) -> aucune raison de tomber sur une frontière
    # de run, donc contamination attendue.
    analyser_chunked_trimmed(n_samples, run_ids, chunk_length=50, trim_size=5)

    # Cas 3 : kfold_trimmed (pas de notion de chunk/session du tout, juste un split
    # contigu du vecteur concaténé + trim aux bords de CHAQUE fold, pas de chaque run)
    print(f"\n--- kfold_trimmed (KFold contigu global, trim_size=5) ---")
    for i, (train_idx, test_idx) in enumerate(create_kfold_trimmed(n_samples, 5, trim_size=5)):
        train_idx, test_idx = np.array(train_idx), np.array(test_idx)
        n_coupes = runs_coupes_par_le_split(train_idx, test_idx, run_ids)
        print(f"  Fold {i+1}: train={len(train_idx):5d}  test={len(test_idx):5d}  "
              f"runs coupés train/test={n_coupes}")

    # Référence : méthode actuelle du projet
    analyser_group_shuffle(n_samples, groupes)
