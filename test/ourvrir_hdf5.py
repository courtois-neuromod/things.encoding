import h5py
import numpy as np

# Remplace par le chemin de ton fichier
chemin_fichier = '../output/hdf5/sub-03.h5'


def explorer_hdf5(chemin):
    """
    Ouvre et explore un fichier HDF5 pour en comprendre la structure.
    """
    print(f"--- Ouverture du fichier : {chemin} ---")

    try:
        with h5py.File(chemin, 'r') as fichier:

            # 1. Lister les "Groupes" (comme des dossiers) à la racine du fichier
            print("\n1. Contenu à la racine :")
            cles_racine = list(fichier.keys())
            print(f"Clés trouvées : {cles_racine}")

            # 2. Explorer récursivement tout le fichier
            print("\n2. Arborescence complète :")

            def imprimer_structure(nom, objet):
                # Si c'est un groupe (un dossier)
                if isinstance(objet, h5py.Group):
                    print(f"Groupe : {nom}")
                # Si c'est un dataset (les données réelles, comme un tableau numpy)
                elif isinstance(objet, h5py.Dataset):
                    print(f"   Dataset : {nom} | Forme (shape) : {objet.shape} | Type : {objet.dtype}")

            # visititems parcourt tout le fichier et applique la fonction à chaque élément
            fichier.visititems(imprimer_structure)

            dataset_exemple = None

            def trouver_premier_dataset(nom, objet):
                nonlocal dataset_exemple
                if isinstance(objet, h5py.Dataset) and dataset_exemple is None:
                    dataset_exemple = nom

            fichier.visititems(trouver_premier_dataset)

            if dataset_exemple:
                print(f"\n3. Lecture du dataset '{dataset_exemple}' :")

                # Accéder au dataset (il agit comme un pointeur vers le disque, pas encore en RAM)
                pointeur_donnees = fichier[dataset_exemple]

                # Pour charger vraiment les données en RAM (en tableau NumPy), on utilise [:]
                donnees_numpy = pointeur_donnees[:]

                print(f"Les 5 premières valeurs :\n{donnees_numpy[:5]}")
                print(f"La taille totale est de : {donnees_numpy.shape}")

                # On peut aussi lire les métadonnées (attributs) cachées dans le dataset
                attributs = list(pointeur_donnees.attrs.keys())
                if attributs:
                    print(f"Métadonnées associées : {attributs}")
                    for attr in attributs:
                        print(f"  - {attr} : {pointeur_donnees.attrs[attr]}")

    except FileNotFoundError:
        print(f"Erreur : Le fichier {chemin} est introuvable.")
    except Exception as e:
        print(f"Une erreur est survenue : {e}")


# Lancer la fonction
explorer_hdf5(chemin_fichier)