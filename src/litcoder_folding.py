"""Port de la méthode de folding de litcoder_core (GT-LIT-Lab), pour test/comparaison
avec les splits maison du projet (`GroupShuffleSplitSession`, `_generer_folds_one_cycle`).

Source : https://github.com/GT-LIT-Lab/litcoder_core/blob/744382bbac3bd614bf286eba6a93e1379c093eca/encoding/models/folding.py

Port quasi verbatim (signatures et logique inchangées) ; seul `print(...)` de debug
retiré et `random.shuffle` remplacé par un `numpy.random.Generator` pour un tirage
reproductible via seed explicite.

Unique ajout du projet : le paramètre optionnel `runs` de
`create_chunked_folds_trimmed`, qui fait porter les chunks sur les runs réels plutôt
que sur des blocs de N échantillons. Sans ce paramètre, le comportement reste
strictement celui de la source (les autres fonctions ne sont pas modifiées).
"""

import numpy as np
from sklearn.model_selection import GroupKFold, KFold, TimeSeriesSplit


def create_folds(
    n_samples: int,
    fold_type: str,
    n_folds: int,
    chunk_length: int | None = None,
    trim_size: int | None = None,
    groups: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
    runs: np.ndarray | None = None,
) -> list[tuple[list[int], list[int]]]:
    """Voir folding.py original. `rng` remplace le `random` global du module source ;
    `runs` est l'ajout du projet, transmis à `create_chunked_folds_trimmed` (ignoré
    par les autres types de folding)."""
    if fold_type == "chunked":
        return create_chunked_folds(
            n_samples, n_folds, chunk_length, shuffle=True, rng=rng
        )
    elif fold_type == "chunked_trimmed":
        if trim_size is None:
            trim_size = 5
        return create_chunked_folds_trimmed(
            n_samples,
            n_folds,
            chunk_length,
            trim_size,
            shuffle=True,
            rng=rng,
            runs=runs,
        )
    elif fold_type == "chunked_contiguous":
        return create_chunked_folds(
            n_samples, n_folds, chunk_length, shuffle=False, rng=rng
        )
    elif fold_type == "kfold":
        kf = KFold(n_splits=n_folds, shuffle=False)
        return list(kf.split(range(n_samples)))
    elif fold_type == "kfold_trimmed":
        if trim_size is None:
            trim_size = 5
        return create_kfold_trimmed(n_samples, n_folds, trim_size)
    elif fold_type == "timeseries":
        tscv = TimeSeriesSplit(n_splits=n_folds)
        return list(tscv.split(range(n_samples)))
    elif fold_type == "group":
        if groups is None:
            raise ValueError("Groups must be provided for group folding")
        gkf = GroupKFold(n_splits=n_folds)
        return list(gkf.split(range(n_samples), groups=groups))
    else:
        raise ValueError(f"Unknown folding type: {fold_type}")


def create_chunked_folds(
    n_samples: int,
    n_folds: int,
    chunk_length: int,
    shuffle: bool = True,
    rng: np.random.Generator | None = None,
) -> list[tuple[list[int], list[int]]]:
    n_complete_chunks = n_samples // chunk_length
    chunk_indices = list(range(n_complete_chunks))

    if shuffle:
        rng = rng or np.random.default_rng()
        rng.shuffle(chunk_indices)

    chunks_per_fold = n_complete_chunks // n_folds
    if chunks_per_fold == 0:
        kf = KFold(n_splits=n_folds, shuffle=shuffle)
        return list(kf.split(range(n_samples)))

    splits = []
    for i in range(n_folds):
        start_idx = i * chunks_per_fold
        end_idx = (i + 1) * chunks_per_fold if i < n_folds - 1 else n_complete_chunks
        test_chunks = chunk_indices[start_idx:end_idx]
        train_chunks = [c for c in chunk_indices if c not in test_chunks]

        test_indices = []
        for chunk in test_chunks:
            start = chunk * chunk_length
            end = start + chunk_length
            test_indices.extend(range(start, min(end, n_samples)))

        train_indices = []
        for chunk in train_chunks:
            start = chunk * chunk_length
            end = start + chunk_length
            train_indices.extend(range(start, min(end, n_samples)))

        splits.append((train_indices, test_indices))

    return splits


def _bornes_des_runs(runs: np.ndarray) -> list[tuple[int, int]]:
    """Délimite les runs contigus d'un tableau d'identifiants par échantillon
    (ex. ["ses-001/run-1", "ses-001/run-1", ..., "ses-001/run-2", ...]).

    Les bornes sont déduites des CHANGEMENTS de valeur, jamais d'une longueur
    supposée constante : après alignement temporel, la longueur d'un run dépend de
    la durée de sa vidéo et n'est donc pas garantie identique partout.

    Returns :
        list[(debut, fin)] : bornes demi-ouvertes, dans l'ordre d'acquisition.
    """
    runs = np.asarray(runs)
    ruptures = np.where(runs[1:] != runs[:-1])[0] + 1
    debuts = np.concatenate(([0], ruptures))
    fins = np.concatenate((ruptures, [len(runs)]))
    return [(int(d), int(f)) for d, f in zip(debuts, fins, strict=True)]


def create_chunked_folds_trimmed(
    n_samples: int,
    n_folds: int,
    chunk_length: int | None = None,
    trim_size: int = 5,
    shuffle: bool = True,
    rng: np.random.Generator | None = None,
    runs: np.ndarray | None = None,
) -> list[tuple[list[int], list[int]]]:
    """Découpe la timeline concaténée en chunks, en teste un sous-ensemble par fold
    (partition : chaque chunk est en test exactement une fois), et rogne `trim_size`
    échantillons aux deux bords de chaque chunk de TEST (les chunks de train restent
    entiers).

    Args :
        runs : identifiant de run par échantillon (ex. `_runs_par_echantillon` de
            RidgeRegression). Si fourni, les chunks sont les RUNS RÉELS et
            `chunk_length` est ignoré ; c'est la seule façon de garantir qu'aucun run
            ne soit coupé entre train et test. Si None (défaut), on retombe sur le
            découpage positionnel de litcoder_core : des blocs de `chunk_length`
            échantillons, dont les frontières ne tombent sur un bord de run que si
            tous les runs font exactement `chunk_length` — pari fragile, un seul run
            plus court décale tous les chunks suivants.
    """
    if runs is not None:
        # `n_samples` n'est plus utilisé pour construire les bornes en mode run : si
        # les deux ne concordent pas, les index renvoyés sortiraient de X/Y sans que
        # rien ne le signale (ou pire, décaleraient silencieusement les folds).
        if len(runs) != n_samples:
            raise ValueError(
                f"runs contient {len(runs)} échantillons mais n_samples={n_samples} : "
                "le tableau de runs doit être aligné ligne à ligne sur X/Y."
            )
        bornes = _bornes_des_runs(runs)
    else:
        if chunk_length is None:
            raise ValueError("chunk_length est requis quand `runs` n'est pas fourni")
        n_complete_chunks = n_samples // chunk_length
        bornes = [
            (c * chunk_length, min((c + 1) * chunk_length, n_samples))
            for c in range(n_complete_chunks)
        ]

    n_chunks = len(bornes)
    chunk_indices = list(range(n_chunks))

    if shuffle:
        rng = rng or np.random.default_rng()
        rng.shuffle(chunk_indices)

    chunks_per_fold = n_chunks // n_folds
    if chunks_per_fold == 0:
        # Repli de litcoder_core : il annule chunking ET trimming. Acceptable sur le
        # chemin d'origine (comportement de référence à préserver), inacceptable en
        # mode run où il ferait silencieusement sauter la protection des frontières.
        if runs is not None:
            raise ValueError(
                f"n_folds={n_folds} > {n_chunks} runs disponibles : impossible de "
                "construire des folds par run."
            )
        kf = KFold(n_splits=n_folds, shuffle=False)
        return list(kf.split(range(n_samples)))

    splits = []
    for i in range(n_folds):
        start_idx = i * chunks_per_fold
        end_idx = (i + 1) * chunks_per_fold if i < n_folds - 1 else n_chunks
        test_chunks = chunk_indices[start_idx:end_idx]
        train_chunks = [c for c in chunk_indices if c not in test_chunks]

        test_indices = []
        for chunk in test_chunks:
            chunk_start, chunk_end = bornes[chunk]
            trimmed_start = chunk_start + trim_size
            trimmed_end = chunk_end - trim_size
            if trimmed_start < trimmed_end:
                test_indices.extend(range(trimmed_start, trimmed_end))

        train_indices = []
        for chunk in train_chunks:
            start, end = bornes[chunk]
            train_indices.extend(range(start, end))

        splits.append((train_indices, test_indices))

    return splits


def create_kfold_trimmed(
    n_samples: int,
    n_folds: int,
    trim_size: int = 5,
) -> list[tuple[list[int], list[int]]]:
    kf = KFold(n_splits=n_folds, shuffle=False)
    base_splits = list(kf.split(range(n_samples)))

    trimmed_splits = []
    for train_indices, test_indices in base_splits:
        train_indices = list(train_indices)
        test_indices = list(test_indices)

        if len(test_indices) > 2 * trim_size:
            trimmed_test = test_indices[trim_size:-trim_size]
        else:
            trimmed_test = test_indices

        trimmed_splits.append((train_indices, trimmed_test))

    return trimmed_splits
