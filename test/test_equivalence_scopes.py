"""Vérifie l'hypothèse dont dépend toute la mécanique des scopes.

Le pipeline ne lance plus la validation croisée qu'UNE fois, sur tout le cerveau, puis
déduit les résultats de chaque scope (parcelles visuelles, ROIs...) par simple sélection
de colonnes — au lieu de relancer une CV par scope.

C'est légitime parce qu'une Ridge multi-cible régresse chaque voxel indépendamment :
`StandardScaler` normalise colonne par colonne, `alpha_per_target=True` choisit un alpha
par cible, `r2_score(multioutput='raw_values')` score par colonne, et le découpage
train/test ne dépend que des sessions/runs — jamais du masque, qui ne filtre que des
colonnes de Y.

Si une version future de scikit-learn couplait les cibles entre elles (par exemple en
partageant un alpha ou une normalisation globale), les figures deviendraient fausses en
silence. D'où ce test.

Aucune donnée réelle n'est nécessaire : X et Y sont synthétiques, seule la mécanique
des modèles est en jeu.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from RidgeRegression import ResultatsCV, masque_relatif, scope_disponible

np.seterr(all="ignore")
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TOLERANCE = 1e-12
echecs = []


def verifier(condition, intitule, detail=""):
    print(
        f"  {'OK   ' if condition else 'ECHEC'} {intitule}{'  ' + detail if detail else ''}"
    )
    if not condition:
        echecs.append(intitule)


# --------------------------------------------------------------------------------
# 1. L'équivalence elle-même, sur les deux chemins de modèle du projet
# --------------------------------------------------------------------------------
def _modele_ridgecv(X, Y, tr, te, alphas):
    """Chemin de `nested_cross_validation_ridgecv_loo` / `chunked_trimmed`."""
    modele = TransformedTargetRegressor(
        regressor=make_pipeline(
            StandardScaler(),
            RidgeCV(alphas=alphas, alpha_per_target=True, cv=None, scoring="r2"),
        ),
        transformer=StandardScaler(),
    )
    modele.fit(X[tr], Y[tr])
    alphas_retenus = modele.regressor_.named_steps["ridgecv"].alpha_
    return r2_score(
        Y[te], modele.predict(X[te]), multioutput="raw_values"
    ), alphas_retenus


def _modele_one_cycle(X, Y, tr, te, alphas):
    """Chemin de `nested_cross_validation_one_cycle` : Ridge manuel + argmax par cible."""
    scaler_X, scaler_Y = StandardScaler(), StandardScaler()
    X_tr = scaler_X.fit_transform(X[tr])
    X_te = scaler_X.transform(X[te])
    Y_tr = scaler_Y.fit_transform(Y[tr])

    r2_par_alpha = np.array(
        [
            r2_score(
                Y[te],
                scaler_Y.inverse_transform(
                    Ridge(alpha=a).fit(X_tr, Y_tr).predict(X_te)
                ),
                multioutput="raw_values",
            )
            for a in alphas
        ]
    )
    alpha_optimal = alphas[np.argmax(r2_par_alpha, axis=0)]
    ridge = Ridge(alpha=alpha_optimal).fit(X_tr, Y_tr)
    r2 = r2_score(
        scaler_Y.transform(Y[te]), ridge.predict(X_te), multioutput="raw_values"
    )
    return r2, alpha_optimal


print("\n--- 1. Restreindre Y à un scope ne change pas les résultats ---")
rng = np.random.default_rng(0)
n_echantillons, n_predicteurs, n_cibles = 400, 60, 40
X = rng.normal(size=(n_echantillons, n_predicteurs))
Y = X @ (rng.normal(size=(n_predicteurs, n_cibles)) * 0.3) + rng.normal(
    size=(n_echantillons, n_cibles)
)
train, test = slice(0, 300), slice(300, None)
grille_alphas = np.logspace(-1, 6, 12)

masque_scope = np.zeros(n_cibles, dtype=bool)
masque_scope[rng.choice(n_cibles, 13, replace=False)] = True

for nom, modele in (
    ("RidgeCV(alpha_per_target)", _modele_ridgecv),
    ("one_cycle (Ridge manuel)", _modele_one_cycle),
):
    r2_complet, alphas_complet = modele(X, Y, train, test, grille_alphas)
    r2_scope, alphas_scope = modele(X, Y[:, masque_scope], train, test, grille_alphas)

    ecart = np.abs(r2_complet[masque_scope] - r2_scope).max()
    verifier(ecart < TOLERANCE, f"{nom} : R² identique", f"(écart max {ecart:.2e})")
    verifier(
        np.array_equal(alphas_complet[masque_scope], alphas_scope),
        f"{nom} : alphas identiques",
    )


# --------------------------------------------------------------------------------
# 2. ResultatsCV.restreindre
# --------------------------------------------------------------------------------
print("\n--- 2. ResultatsCV.restreindre ---")
n_folds, n_features = 5, 20
masque = np.zeros(n_features, dtype=bool)
masque[[1, 4, 7, 11, 19]] = True

complet = ResultatsCV(
    r2_moyen=rng.normal(size=n_features),
    r2_variance_inter_folds=rng.normal(size=n_features),
    r2_tous_les_tests=rng.normal(size=(n_folds, n_features)),
    alphas_tous_externes=rng.normal(size=(n_folds, n_features)),
    alphas_tous_externes_moyen=rng.normal(size=n_features),
)
restreint = complet.restreindre(masque)

verifier(
    complet.restreindre(None) is complet, "restreindre(None) renvoie l'objet tel quel"
)
for nom in ResultatsCV._CHAMPS_PAR_FEATURE:
    valeur = getattr(complet, nom)
    if valeur is None:
        verifier(getattr(restreint, nom) is None, f"{nom} : None reste None")
    else:
        verifier(
            np.array_equal(getattr(restreint, nom), valeur[masque]),
            f"{nom} : v[masque]",
        )
for nom in ResultatsCV._CHAMPS_PAR_LIGNE:
    valeur = getattr(complet, nom)
    if valeur is None:
        verifier(getattr(restreint, nom) is None, f"{nom} : None reste None")
    else:
        verifier(
            np.array_equal(getattr(restreint, nom), valeur[:, masque]),
            f"{nom} : v[:, masque]",
        )

# Les champs optionnels doivent être restreints eux aussi quand ils sont présents.
avec_options = ResultatsCV(
    r2_moyen=rng.normal(size=n_features),
    r2_variance_inter_folds=rng.normal(size=n_features),
    r2_tous_les_tests=rng.normal(size=(n_folds, n_features)),
    alphas_tous_externes=rng.normal(size=(n_folds, n_features)),
    alphas_tous_externes_moyen=rng.normal(size=n_features),
    best_alphas_inner=rng.normal(size=(n_folds * 3, n_features)),
    pearson_moyen=rng.normal(size=n_features),
    pearson_variance_inter_folds=rng.normal(size=n_features),
    pearson_tous_les_tests=rng.normal(size=(n_folds, n_features)),
)
opt = avec_options.restreindre(masque)
verifier(
    opt.best_alphas_inner.shape == (n_folds * 3, masque.sum()),
    "best_alphas_inner : seule la 2e dimension est restreinte",
    f"{opt.best_alphas_inner.shape}",
)
verifier(
    np.array_equal(
        opt.pearson_tous_les_tests, avec_options.pearson_tous_les_tests[:, masque]
    ),
    "pearson_tous_les_tests : v[:, masque]",
)

# Garde-fou : tout champ ajouté à la dataclass doit être classé dans l'une des deux
# listes, sinon `restreindre` le perdrait en silence.
champs_declares = set(ResultatsCV._CHAMPS_PAR_FEATURE) | set(
    ResultatsCV._CHAMPS_PAR_LIGNE
)
champs_reels = set(ResultatsCV.__dataclass_fields__)
verifier(
    champs_declares == champs_reels,
    "tous les champs de la dataclass sont classés",
    f"(oubliés : {sorted(champs_reels - champs_declares) or 'aucun'})",
)


# --------------------------------------------------------------------------------
# 3. masque_relatif
# --------------------------------------------------------------------------------
print("\n--- 3. scope_disponible / masque_relatif ---")
espace_complet = 10
masque_cv = np.zeros(espace_complet, dtype=bool)
masque_cv[[0, 2, 4, 6, 8]] = True  # CV sur 5 des 10 unités
inclus = np.zeros(espace_complet, dtype=bool)
inclus[[2, 6]] = True  # sous-ensemble du scope de CV
deborde = np.zeros(espace_complet, dtype=bool)
deborde[[2, 3]] = True  # 3 n'est pas dans le scope de CV

# `scope_disponible` répond « puis-je dériver ce scope ? », `masque_relatif` répond
# « comment ? ». Les confondre revenait à écarter le cerveau entier dans la config par
# défaut (scope_cv=None, masque de scope None) — le défaut que ce bloc verrouille.
verifier(
    scope_disponible(None, None), "CV sur tout + scope cerveau entier : disponible"
)
verifier(scope_disponible(inclus, None), "CV sur tout + scope restreint : disponible")
verifier(
    scope_disponible(inclus, masque_cv), "CV restreinte + scope inclus : disponible"
)
verifier(
    not scope_disponible(deborde, masque_cv),
    "CV restreinte + scope débordant : indisponible",
)
verifier(
    not scope_disponible(None, masque_cv),
    "CV restreinte + scope cerveau entier : indisponible",
    "(sinon on étiquetterait le sous-espace comme complet)",
)

verifier(masque_relatif(None, None) is None, "aucune restriction : renvoie None")
verifier(
    np.array_equal(masque_relatif(inclus, None), inclus),
    "masque_cv=None : le masque passe tel quel",
)
verifier(
    np.array_equal(
        masque_relatif(inclus, masque_cv), np.array([False, True, False, True, False])
    ),
    "scope inclus : reprojection dans l'espace de la CV",
)

# Cohérence de bout en bout : restreindre par le masque reprojeté doit donner les
# mêmes colonnes que restreindre l'espace complet par le masque d'origine.
donnees = rng.normal(size=(3, espace_complet))
verifier(
    np.array_equal(
        donnees[:, masque_cv][:, masque_relatif(inclus, masque_cv)], donnees[:, inclus]
    ),
    "reprojection cohérente avec une sélection directe",
)


print("\n" + ("TOUT PASSE" if not echecs else f"ECHECS ({len(echecs)}) : {echecs}"))
sys.exit(1 if echecs else 0)
