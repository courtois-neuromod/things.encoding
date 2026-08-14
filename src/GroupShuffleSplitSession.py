import numpy as np


class GroupShuffleSplitSession:
    """Tirage aléatoire de sessions entières en test, tout le reste en train.

    Aucun buffer : contrairement à `GroupShuffleSplitRun`, on ne retire PAS du train
    les sessions voisines de celles du test. Entre deux sessions il y a plusieurs
    jours d'écart — ni autocorrélation BOLD, ni drift de scanner commun — donc le
    buffer ne protégeait de rien et coûtait du train. Le buffer d'adjacence a un
    sens au niveau du run (quelques minutes d'écart) : voir `GroupShuffleSplitRun`.

    À noter : ce split reste le plus STRICT des deux vis-à-vis des confusions à
    l'échelle de la session (même jour, même positionnement de tête, même drift
    lent), puisqu'il met la session entière du côté test. `GroupShuffleSplitRun`,
    lui, laisse en train les runs frères de la session testée. Aucun des deux ne
    domine l'autre : cf. la docstring de `GroupShuffleSplitRun` pour la lecture
    correcte d'un écart de R² entre les deux niveaux.

    Conséquence : train et test partitionnent l'ensemble des échantillons
    (`len(train_idx) + len(test_idx) == n_samples` sur chaque fold).
    """

    def __init__(self, n_splits=5, test_size=0.2, train_size=None, random_state=None):
        self.n_splits = n_splits
        self.random_state = random_state
        self.test_size = test_size
        self.train_size = train_size

    def split(self, X, Y=None, groups=None):
        if groups is None:
            raise ValueError("Groups ne peut pas être None")

        sessions_uniques = np.unique(groups)
        n_sessions = len(sessions_uniques)
        rng = np.random.default_rng(self.random_state)

        if self.test_size is not None:
            if isinstance(self.test_size, float):
                nb_test = max(1, round(n_sessions * self.test_size))
            else:
                nb_test = max(1, self.test_size)
            nb_train = n_sessions - nb_test
        elif self.train_size is not None:
            if isinstance(self.train_size, float):
                nb_train = max(1, round(n_sessions * self.train_size))
            else:
                nb_train = max(1, self.train_size)
            nb_test = n_sessions - nb_train
        else:
            nb_test = max(1, round(n_sessions * 0.2))
            nb_train = n_sessions - nb_test

        for _ in range(self.n_splits):
            shuffled = rng.permutation(sessions_uniques)
            test_sessions = shuffled[:nb_test]
            train_sessions = shuffled[nb_test : nb_test + nb_train]

            # Scikit-learn attend les index des lignes (et non les IDs des sessions)
            train_idx = np.where(np.isin(groups, train_sessions))[0]
            test_idx = np.where(np.isin(groups, test_sessions))[0]

            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits
