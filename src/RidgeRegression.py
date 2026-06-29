"""
Régression Ridge pour l'encodage cérébral THINGS memory.
Entraîne une RidgeCV par couche et évalue la prédiction.
"""
from pathlib import Path
from TribeHDF5Normalization import TribeHDF5Normalization
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
import numpy as np
import h5py
from sklearn.model_selection import LeaveOneGroupOut

# Chemins
"""
#Rorqual
ROOT_ENCODING    = Path("/home/aclaud/links/scratch/things.encoding")
ROOT_TIMESERIES  = Path("/home/aclaud/links/scratch/things.timeseries")

SUB   = "sub-01"
LAYER = "encoder_layer7_ffn"

chemin_tribe = ROOT_ENCODING / "output" / "hdf5" / "things_encoding" / f"{SUB}.h5"

chemin_cneuromod = (
    ROOT_TIMESERIES / "timeseries" / "cneuromod2026" / SUB /
    f"{SUB}_task-things_space-MNI152NLin2009cAsym_atlas-cneuromod26_desc-1134Parcels_timeseries.h5"
)
"""

ROOT = Path(__file__).parent.parent

SUB   = "sub-01"
LAYER = "encoder_layer7_ffn"

chemin_tribe = ROOT / "output" / "hdf5" / "sub-01.h5"

chemin_cneuromod = (
    ROOT / "data" / "timeseries" / "cneuromod2026" / SUB /
    f"{SUB}_task-things_space-MNI152NLin2009cAsym_atlas-cneuromod26_desc-1134Parcels_timeseries.h5"
)

# Découverte automatique des runs disponibles dans le HDF5 TRIBE
runs = []
with h5py.File(chemin_tribe, "r") as f:
    for tribe_ses in sorted(f.keys()):           # "ses-001", "ses-002", ...
        for tribe_run in sorted(f[tribe_ses].keys()):   # "run-1", "run-2", ...

            # Conversion ses-001 → ses-01 pour CNeuroMod
            num_ses = int(tribe_ses.replace("ses-", ""))
            cneuromod_ses = f"ses-{num_ses:02d}"

            # Clé dataset CNeuroMod
            num_run = tribe_run.replace("run-", "")
            cneuromod_dataset = f"{cneuromod_ses}_task-things_run-{num_run}_timeseries"

            # Chemin vidéo originale (non CFR) pour ffprobe
            nom_video = f"{SUB}_{tribe_ses}_task-thingsmemory_{tribe_run}.mp4"
            #Rorqual
            #chemin_video = ROOT_ENCODING / "data" / "data" / SUB / tribe_ses / nom_video
            chemin_video = ROOT / "data" / SUB / tribe_ses / nom_video

            runs.append((tribe_ses, tribe_run, chemin_video, cneuromod_ses, cneuromod_dataset))

print(f"{len(runs)} runs trouvés dans {chemin_tribe.name}")

# Alignement temporel et concaténation
X_list, Y_list = [], []
runs_ok = []
groupes_list = []
for id_run, (tribe_ses, tribe_run, chemin_video, cneuromod_ses, cneuromod_dataset) in enumerate(runs):

    # Vérifier que la vidéo source existe localement
    if not chemin_video.exists():
        print(f"Vidéo manquante, run ignoré : {chemin_video.name}")
        continue

    try:
        normalisateur = TribeHDF5Normalization(
            chemin_tribe=chemin_tribe,
            chemin_cneuromod=chemin_cneuromod,
            chemin_video=chemin_video,
            tribe_ses=tribe_ses,
            tribe_run=tribe_run,
            tribe_layer=LAYER,
            cneuromod_ses=cneuromod_ses,
            cneuromod_dataset=cneuromod_dataset,
            t_Tribe_s=0.5,
            TR_irmf_s=1.49,
        )
        X_run, Y_run = normalisateur.executer_pipeline()
        X_list.append(X_run)
        Y_list.append(Y_run)
        runs_ok.append(f"{tribe_ses}/{tribe_run}")
        id_array = np.full(X_run.shape[0], id_run)
        groupes_list.append(id_array)

    except Exception as e:
        print(f"✗ {tribe_ses}/{tribe_run} ignoré — {e}")

print(f"\n{len(runs_ok)} runs traités avec succès")

X = np.concatenate(X_list, axis=0)
Y = np.concatenate(Y_list, axis=0)
groupes = np.concatenate(groupes_list, axis=0)
print(f"Matrice finale : X={X.shape}, Y={Y.shape}")

session_par_run = {}
for id_run, label in enumerate(runs_ok):
    ses = label.split("/")[0]  # "ses-001"
    session_par_run[id_run] = ses

sessions_uniques = sorted(set(session_par_run.values()))

# --- PARAMÈTRES ML ---
alphas = np.logspace(-1, 5, 20)
scores_tous_les_folds = []

print(f"\n{len(sessions_uniques)} sessions")

for ses in sessions_uniques:
    # Trouver les ids de runs appartenant à cette session
    ids_session = [i for i, s in session_par_run.items() if s == ses]

    if len(ids_session) < 2:
        print(f"\n{ses} : pas assez de runs ({len(ids_session)}), ignorée")
        continue

    # Train = tous les runs sauf le dernier, Test = dernier run
    ids_train = ids_session[:-1]
    ids_test  = ids_session[-1:]

    train_mask = np.isin(groupes, ids_train)
    test_mask  = np.isin(groupes, ids_test)

    X_train, Y_train = X[train_mask], Y[train_mask]
    X_test,  Y_test  = X[test_mask],  Y[test_mask]

    print(f"\n--- Session {ses} | Train : {X_train.shape[0]} TRs ({len(ids_train)} runs) | Test : {X_test.shape[0]} TRs (1 run) ---")

    # Standardisation
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    Y_train_scaled = scaler_Y.fit_transform(Y_train)
    X_test_scaled  = scaler_X.transform(X_test)
    Y_test_scaled  = scaler_Y.transform(Y_test)

    # RidgeCV
    modele = RidgeCV(alphas=alphas, alpha_per_target=True)
    modele.fit(X_train_scaled, Y_train_scaled)

    # Score
    Y_pred_scaled = modele.predict(X_test_scaled)
    scores_r2 = r2_score(Y_test_scaled, Y_pred_scaled, multioutput="raw_values")
    scores_tous_les_folds.append(scores_r2)
    print(f"    -> R² max : {np.max(scores_r2):.4f}")

"""
# --- PARAMÈTRES ML ---
alphas = np.logspace(-1, 5, 20)
logo = LeaveOneGroupOut()
scores_tous_les_folds = []

print(f"\n[Validation Croisée] Lancement du LORO-CV sur {len(runs_ok)} runs...")

# Séparation automatique des runs pour le test, et le reste pour le train
for index_fold, (train_index, test_index) in enumerate(logo.split(X, Y, groupes)):
    print(f"\n--- Évaluation du Fold {index_fold + 1}/{len(runs_ok)} ---")

    # 1. Découpage dynamique
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]

    # 2. Standardisation
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    Y_train_scaled = scaler_Y.fit_transform(Y_train)
    X_test_scaled = scaler_X.transform(X_test)
    Y_test_scaled = scaler_Y.transform(Y_test)

    # 3. Entraînement RidgeCV
    modele = RidgeCV(alphas=alphas, alpha_per_target=True)
    modele.fit(X_train_scaled, Y_train_scaled)

    # 4. Prédiction et Score de ce Fold
    Y_pred_scaled = modele.predict(X_test_scaled)
    scores_r2 = r2_score(Y_test_scaled, Y_pred_scaled, multioutput="raw_values")

    scores_tous_les_folds.append(scores_r2)
    print(f"    -> R² max sur ce fold : {np.max(scores_r2):.4f}")
"""
# --- RÉSULTATS GLOBAUX ---
scores_finaux = np.mean(scores_tous_les_folds, axis=0)

score_moyen = np.mean(scores_finaux)
score_median = np.median(scores_finaux)
score_max = np.max(scores_finaux)
parcelle_max = np.argmax(scores_finaux)
n_positifs = np.sum(scores_finaux > 0)

print(f"\n=========================================")
print(f"[Résultats Finaux Robustes — couche {LAYER}]")
print(f"R² moyen   : {score_moyen:.4f}")
print(f"R² médian  : {score_median:.4f}")
print(f"R² max     : {score_max:.4f}  (parcelle {parcelle_max})")
print(f"Parcelles R² > 0 : {n_positifs} / {len(scores_finaux)}")
print(f"=========================================")
