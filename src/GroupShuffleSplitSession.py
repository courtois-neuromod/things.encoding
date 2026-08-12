import numpy as np


class GroupShuffleSplitSession:

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
            train_sessions = shuffled[nb_test:nb_test + nb_train]

            # Scikit-learn attend les index des lignes (et non les IDs des sessions)
            train_idx = np.where(np.isin(groups, train_sessions))[0]
            test_idx = np.where(np.isin(groups, test_sessions))[0]

            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits