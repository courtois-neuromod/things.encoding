"""Compare les découpages externes disponibles sur la vraie structure session/run
de sub-01 :
  - `create_chunked_folds_trimmed` (litcoder_core, cf. src/litcoder_folding.py), en
    mode TR (chunk_length) et en mode run (runs=...) ;
  - `GroupShuffleSplitSession` (tirage de sessions, sans buffer) ;
  - `GroupShuffleSplitRun` (LORO aléatoire, runs adjacents exclus du train).

Le critère central est le nombre de RUNS COUPÉS, c'est-à-dire les runs dont des TR
se retrouvent à la fois en train et en test : c'est la fuite que le découpage par
run élimine par construction.

Note : le fichier TRIBE (output/features/things_encoding/sub-01.h5, 14 Go) est un
placeholder iCloud "dataless" sur cette machine (0 octet réellement sur disque,
cf. `ls -lO` -> flag "dataless") : toute lecture déclenche un téléchargement qui
bloque indéfiniment. On n'a donc pas besoin de X ici : ce script ne teste que la
LOGIQUE DE SPLIT (indices de TR), donc on prend n_samples/groupes/run_ids
directement depuis le fichier BOLD (petit, ~170 Mo, déjà local), sans passer par
RidgeRegression.create_X_Y_total().

Conséquence : les longueurs lues sont les longueurs BRUTES (190 TR/run), pas celles
d'après alignement (~180 TR), et les runs écartés par create_X_Y_total (vidéo
manquante, dataset BOLD absent) sont ici présents. Ça valide la logique de split,
pas les tailles exactes du pipeline.
"""

import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from GroupShuffleSplitRun import GroupShuffleSplitRun
from GroupShuffleSplitSession import GroupShuffleSplitSession
from litcoder_folding import create_chunked_folds_trimmed, create_kfold_trimmed

CHEMIN_BOLD = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "timeseries"
    / "cneuromod2026"
    / "sub-01"
    / "sub-01_task-things_space-MNI152NLin2009cAsym_atlas-cneuromod26_desc-1134Parcels_timeseries.h5"
)


def charger_structure_reelle(chemin_bold: Path):
    """Reproduit l'ordre de concaténation de RidgeRegression.create_X_Y_total()
    (sessions triées, runs triés) et renvoie n_samples, groupes (session), run_ids
    (identifiant global de run, plus fin que la session) et runs_labels (les mêmes
    chaînes "ses-XXX/run-Y" que RidgeRegression._runs_par_echantillon)."""
    groupes, run_ids, runs_labels = [], [], []
    run_id_global = 0
    with h5py.File(chemin_bold, "r") as f:
        for ses in sorted(f.keys()):
            num_ses = int(ses.replace("ses-", ""))
            for run_ds in sorted(f[ses].keys()):
                n_tr = f[ses][run_ds].shape[0]
                # "ses-01_task-things_run-3_timeseries" -> "3"
                num_run = run_ds.rsplit("run-", 1)[1].split("_")[0]
                groupes.append(np.full(n_tr, num_ses))
                run_ids.append(np.full(n_tr, run_id_global))
                runs_labels.append(np.full(n_tr, f"ses-{num_ses:03d}/run-{num_run}"))
                run_id_global += 1
    groupes = np.concatenate(groupes)
    run_ids = np.concatenate(run_ids)
    runs_labels = np.concatenate(runs_labels)
    return len(groupes), groupes, run_ids, runs_labels


def runs_coupes_par_le_split(train_idx, test_idx, run_ids):
    """Nombre de runs dont des TR se retrouvent à la fois en train ET en test
    (= contamination : le split ignore la frontière de run)."""
    runs_train = set(run_ids[train_idx])
    runs_test = set(run_ids[test_idx])
    return len(runs_train & runs_test)


def runs_voisins_en_train(train_idx, test_idx, run_ids):
    """Nombre de runs de train immédiatement voisins (avant/après dans l'ordre
    d'acquisition) d'un run de test. Doit valoir 0 quand le buffer est actif."""
    runs_train = set(run_ids[train_idx])
    voisins = set()
    for r in set(run_ids[test_idx]):
        voisins.update({r - 1, r + 1})
    return len(runs_train & voisins)


def analyser_chunked(
    n_samples,
    run_ids,
    n_folds=5,
    trim_size=5,
    seed=0,
    chunk_length=None,
    runs=None,
    titre="",
):
    folds = create_chunked_folds_trimmed(
        n_samples,
        n_folds,
        chunk_length,
        trim_size=trim_size,
        shuffle=True,
        rng=np.random.default_rng(seed),
        runs=runs,
    )
    print(f"\n--- {titre} | trim_size={trim_size} | n_folds={n_folds} ---")
    runs_testes = []
    for i, (train_idx, test_idx) in enumerate(folds):
        train_idx, test_idx = np.array(train_idx), np.array(test_idx)
        n_coupes = runs_coupes_par_le_split(train_idx, test_idx, run_ids)
        runs_testes.extend(set(run_ids[test_idx]))
        print(
            f"  Fold {i + 1}: train={len(train_idx):5d}  test={len(test_idx):5d}  "
            f"runs coupés train/test={n_coupes}"
        )
    print(
        f"  -> runs testés au total : {len(runs_testes)} (doublons : "
        f"{len(runs_testes) - len(set(runs_testes))})"
    )


def analyser_splitter(
    splitter, groupes_split, run_ids, n_samples, titre, buffer_attendu
):
    print(f"\n--- {titre} ---")
    for i, (train_idx, test_idx) in enumerate(
        splitter.split(None, None, groupes_split)
    ):
        n_coupes = runs_coupes_par_le_split(train_idx, test_idx, run_ids)
        n_voisins = runs_voisins_en_train(train_idx, test_idx, run_ids)
        ecartes = n_samples - len(train_idx) - len(test_idx)
        print(
            f"  Fold {i + 1}: train={len(train_idx):5d} ({len(set(run_ids[train_idx])):3d} runs)  "
            f"test={len(test_idx):5d} ({len(set(run_ids[test_idx])):3d} runs)  "
            f"écartés={ecartes:5d}  runs coupés={n_coupes}  voisins en train={n_voisins}"
        )
    if buffer_attendu:
        print("  -> attendu : voisins en train = 0, écartés > 0 (buffer actif)")
    else:
        print("  -> attendu : écartés = 0 (aucun buffer, train+test = tous les TR)")


if __name__ == "__main__":
    n_samples, groupes, run_ids, runs_labels = charger_structure_reelle(CHEMIN_BOLD)
    n_sessions = len(set(groupes))
    n_runs = len(set(run_ids))
    print(
        f"sub-01 : n_samples={n_samples}  n_sessions={n_sessions}  n_runs={n_runs}  "
        f"(chaque run = 190 TR bruts, exactement, vérifié sur les {n_runs} runs)"
    )

    # Cas 1 : découpage par RUN réel -> les chunks sont les runs, donc aucun run ne
    # peut être coupé, quelle que soit sa longueur.
    analyser_chunked(
        n_samples,
        run_ids,
        runs=runs_labels,
        titre="chunked_trimmed | chunks = RUNS réels (runs=...)",
    )

    # Cas 2 : chunk_length aligné sur la longueur d'un run (190 TR) -> les chunks ne
    # tombent pas à cheval sur deux runs ici, mais uniquement parce que tous les runs
    # font exactement 190 TR dans le fichier BRUT. Après alignement ce n'est plus
    # garanti : c'est précisément ce que le cas 1 rend robuste.
    analyser_chunked(
        n_samples,
        run_ids,
        chunk_length=190,
        titre="chunked_trimmed | chunk_length=190 (= 1 run brut)",
    )

    # Cas 3 : chunk_length "naïf" (ex. 50 TR, valeur typique vue dans des configs
    # litcoder pour des scans continus) -> aucune raison de tomber sur une frontière
    # de run, donc contamination attendue.
    analyser_chunked(
        n_samples,
        run_ids,
        chunk_length=50,
        titre="chunked_trimmed | chunk_length=50 (naïf)",
    )

    # Cas 4 : kfold_trimmed (pas de notion de chunk/session du tout, juste un split
    # contigu du vecteur concaténé + trim aux bords de CHAQUE fold, pas de chaque run)
    print("\n--- kfold_trimmed (KFold contigu global, trim_size=5) ---")
    for i, (train_idx, test_idx) in enumerate(
        create_kfold_trimmed(n_samples, 5, trim_size=5)
    ):
        train_idx, test_idx = np.array(train_idx), np.array(test_idx)
        n_coupes = runs_coupes_par_le_split(train_idx, test_idx, run_ids)
        print(
            f"  Fold {i + 1}: train={len(train_idx):5d}  test={len(test_idx):5d}  "
            f"runs coupés train/test={n_coupes}"
        )

    # Cas 5 : tirage par session, sans buffer (le buffer d'adjacence a été retiré :
    # entre deux sessions il y a plusieurs jours, il ne protégeait de rien).
    analyser_splitter(
        GroupShuffleSplitSession(n_splits=5, test_size=0.1, random_state=0),
        groupes,
        run_ids,
        n_samples,
        "GroupShuffleSplitSession | test_size=0.1 (niveau session, sans buffer)",
        buffer_attendu=False,
    )

    # Cas 6 : LORO aléatoire, plusieurs runs par fold, avec buffer.
    analyser_splitter(
        GroupShuffleSplitRun(n_splits=5, test_size=0.1, random_state=0),
        runs_labels,
        run_ids,
        n_samples,
        "GroupShuffleSplitRun | test_size=0.1 (21 runs de test attendus)",
        buffer_attendu=True,
    )

    # Cas 7 : LORO strict — 1 seul run en test, 2 runs écartés en buffer.
    analyser_splitter(
        GroupShuffleSplitRun(n_splits=5, test_size=1, random_state=0),
        runs_labels,
        run_ids,
        n_samples,
        "GroupShuffleSplitRun | test_size=1 (LORO strict)",
        buffer_attendu=True,
    )

    # Cas 8 : contrôle du coût du buffer. Même tirage qu'au cas 6, n_buffer=0 : le
    # train retrouve la taille qu'il a au niveau session (192 runs), donc un écart de
    # R² entre le cas 8 et le cas 5 ne peut plus s'expliquer par le volume de données
    # — il vient des runs frères de la session testée, restés en train ici.
    analyser_splitter(
        GroupShuffleSplitRun(n_splits=5, test_size=0.1, random_state=0, n_buffer=0),
        runs_labels,
        run_ids,
        n_samples,
        "GroupShuffleSplitRun | test_size=0.1, n_buffer=0 (contrôle : voisins en train attendus)",
        buffer_attendu=False,
    )

    # Cas 9 : garde-fous (doivent lever, pas passer silencieusement).
    print("\n--- Garde-fous ---")
    for description, appel in [
        (
            "runs désaligné sur n_samples",
            lambda: create_chunked_folds_trimmed(
                n_samples + 1, 5, None, runs=runs_labels
            ),
        ),
        (
            "n_folds > nombre de runs en mode run",
            lambda: create_chunked_folds_trimmed(
                n_samples, n_runs + 1, None, runs=runs_labels
            ),
        ),
        # n_runs-1 runs en test : l'unique run de train est forcément voisin d'un run
        # de test, donc le buffer le retire et le train est vide.
        (
            "buffer vidant le train",
            lambda: list(
                GroupShuffleSplitRun(
                    n_splits=1, test_size=n_runs - 1, random_state=0
                ).split(None, None, runs_labels)
            ),
        ),
    ]:
        try:
            appel()
            print(f"  {description:45s} -> PAS D'ERREUR (attendu : ValueError)")
        except ValueError as e:
            print(f"  {description:45s} -> ValueError OK ({str(e)[:60]}...)")
