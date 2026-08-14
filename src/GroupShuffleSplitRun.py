import numpy as np


class GroupShuffleSplitRun:
    """Leave-One-Run-Out aléatoire : tirage de runs entiers en test, avec exclusion
    optionnelle des runs adjacents.

    Même mécanique que `GroupShuffleSplitSession` (mêmes conventions scikit-learn :
    `groups` donne un identifiant par échantillon, `split` renvoie des index de
    lignes), mais le groupe est le RUN — l'unité d'acquisition élémentaire.

    Différence de fond : ici les runs immédiatement adjacents au test peuvent être
    retirés du train (`n_buffer`). Deux runs consécutifs ne sont séparés que de
    quelques minutes (même passage dans le scanner, même drift, même état
    physiologique), contrairement à deux sessions séparées de plusieurs jours —
    c'est à ce niveau que le buffer a un sens. L'adjacence est comprise dans l'ordre
    d'acquisition global : pour un run de début/fin de session on écarte aussi un run
    distant de plusieurs jours, ce qui est inutile mais inoffensif.

    ATTENTION — ce split n'est PAS « plus propre » que le split par session, il est
    plus propre sur un axe et plus sale sur un autre :
        - vs session : le buffer supprime la contiguïté temporelle immédiate, que le
          split par session laissait passer (un run de test pouvait suivre de 6 min
          un run de train de la session voisine — mais ça reste marginal, quelques
          runs par fold, cf. test/test_litcoder_folding.py).
        - vs session, dans l'autre sens : les AUTRES runs de la session testée (2 à 5
          runs, même jour, même repositionnement de tête, même drift lent, même état
          du sujet) restent en train. Le split par session les excluait tous. Au
          niveau de la session, ce split fuit donc PLUS, pas moins.
    Conséquence pratique : un R² plus haut au niveau run qu'au niveau session ne
    veut pas dire « meilleur modèle », et un R² plus bas ne veut pas dire « moins de
    fuite ». Deux effets tirent en sens opposés — moins de train (buffer) vers le
    bas, runs frères de la même session vers le haut. Pour les séparer, comparer
    d'abord `n_buffer=1` et `n_buffer=0` à niveau_split="run" : à n_buffer=0 le train
    retrouve sa taille du niveau session, donc l'écart restant est purement l'effet
    « runs frères ».

    `test_size` donne les deux régimes utiles :
        - int   : nombre exact de runs en test. `test_size=1` = LORO strict
                  (1 run testé, 2 runs écartés en buffer, tous les autres en train).
        - float : proportion des runs. Sur 213 runs, `test_size=0.1` donne 21 runs de
                  test et 30 à 40 runs de buffer, soit un train de 152 à 158 runs
                  (~72 %) au lieu de 192 (90 %) — mesuré, cf. test_litcoder_folding.

    Attention : dès que `n_buffer > 0`, train et test ne partitionnent PAS les
    échantillons (`len(train_idx) + len(test_idx) < n_samples`).
    """

    def __init__(
        self, n_splits=5, test_size=0.2, train_size=None, random_state=None, n_buffer=1
    ):
        self.n_splits = n_splits
        self.random_state = random_state
        self.test_size = test_size
        self.train_size = train_size
        self.n_buffer = n_buffer

    def split(self, X, Y=None, groups=None):
        if groups is None:
            raise ValueError("Groups ne peut pas être None")

        groups = np.asarray(groups)

        # Ordre d'ACQUISITION, pas ordre trié : `np.unique` seul trierait
        # lexicographiquement des identifiants de run ("ses-014/run-3"), ce qui
        # marche par chance ici (sessions sur 3 chiffres, runs 1..6, non zéro-paddés)
        # mais donnerait un voisinage faux dès qu'un numéro de run passe à deux
        # chiffres. Les runs étant contigus dans X/Y par construction
        # (`create_X_Y_total` concatène dans l'ordre), l'ordre de première apparition
        # est l'ordre d'acquisition réel.
        _, premieres_positions = np.unique(groups, return_index=True)
        runs_uniques = groups[np.sort(premieres_positions)]
        n_runs = len(runs_uniques)

        # Position dans l'ordre d'acquisition, pour retrouver les voisins d'un run.
        position_de = {run: i for i, run in enumerate(runs_uniques)}

        rng = np.random.default_rng(self.random_state)

        if self.test_size is not None:
            if isinstance(self.test_size, float):
                nb_test = max(1, round(n_runs * self.test_size))
            else:
                nb_test = max(1, self.test_size)
            nb_train = n_runs - nb_test
        elif self.train_size is not None:
            if isinstance(self.train_size, float):
                nb_train = max(1, round(n_runs * self.train_size))
            else:
                nb_train = max(1, self.train_size)
            nb_test = n_runs - nb_train
        else:
            nb_test = max(1, round(n_runs * 0.2))
            nb_train = n_runs - nb_test

        if nb_test >= n_runs:
            raise ValueError(
                f"test_size={self.test_size} met {nb_test} runs en test sur {n_runs} "
                "disponibles : il ne resterait rien en train."
            )

        for i_fold in range(self.n_splits):
            shuffled = rng.permutation(runs_uniques)
            test_runs = shuffled[:nb_test]
            train_runs_raw = shuffled[nb_test : nb_test + nb_train]

            # Voisins à +/- n_buffer de chaque run de test, dans l'ordre d'acquisition.
            adjacents = set()
            for run_test in test_runs:
                idx = position_de[run_test]
                for decalage in range(1, self.n_buffer + 1):
                    if idx - decalage >= 0:
                        adjacents.add(runs_uniques[idx - decalage])
                    if idx + decalage < n_runs:
                        adjacents.add(runs_uniques[idx + decalage])

            train_runs = [r for r in train_runs_raw if r not in adjacents]

            # Le buffer peut vider le train si test_size est grand (chaque run de test
            # en écarte jusqu'à 2*n_buffer) : mieux vaut une erreur explicite ici
            # qu'un `Ridge.fit` sur 0 échantillon plus loin.
            if not train_runs:
                raise ValueError(
                    f"Fold {i_fold + 1} : le buffer (n_buffer={self.n_buffer}) a retiré "
                    f"tous les runs de train ({nb_test} runs en test sur {n_runs}). "
                    "Réduire test_size ou n_buffer."
                )

            # Scikit-learn attend les index des lignes (et non les IDs des runs)
            train_idx = np.where(np.isin(groups, train_runs))[0]
            test_idx = np.where(np.isin(groups, test_runs))[0]

            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits
