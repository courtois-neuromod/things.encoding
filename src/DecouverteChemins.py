import os
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path

# Les chemins ignorés globalement
DOSSIERS_IGNORES = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        "node_modules",
        ".datalad",
        ".ruff_cache",
        ".pytest_cache",
        "site-packages",
        ".mypy_cache",
    }
)
TACHES_CONNUES = ("things", "friends")
NOM_FICHIER_ANNOTATIONS_PARCELLES = "tpl-MNI152NLin2009cAsym_atlas-Schaefer2018TianS3NettekovenAsym_desc-1000Parcels7Networks50Subcort128Cereb_parcelAnnotations.tsv"


@dataclass
class CheminsProjet:
    """Regroupe tous les chemins de fichiers nécessaires pour un sujet donné."""

    root_encoding: Path
    racines_recherche: tuple
    taches: str
    dossier_embeddings: Path
    chemin_tribe: Path
    chemin_cneuromod: Path
    chemin_atlas: Path
    chemin_ROImask: Path
    videos_par_nom: dict = None
    chemin_anatomie: Path = None
    chemin_annotations_parcelles: Path = None


@dataclass
class UniteAlignement:
    """Une unité alignable : un run Things, ou un épisode Friends."""

    chemin_tribe: Path
    cles_tribe: tuple
    cneuromod_ses: str
    cneuromod_dataset: str
    chemin_video: Path
    num_session: int
    identifiant: str


class DecouverteChemins:
    """Gestionnaire de découverte dynamique des chemins de fichiers et dossiers du projet."""

    def __init__(self, subject, taches, flag_precision_voxel, racines_recherche=None):
        self.subject = subject
        self.taches = taches
        self.flag_precision_voxel = flag_precision_voxel
        self.racines_recherche = racines_recherche

        self._chemins = None
        self._index = None

    @property
    def chemins(self):
        if self._chemins is None:
            self._chemins = self.get_path_file()
        return self._chemins

    def _racines_recherche(self):
        if self.racines_recherche is not None:
            return tuple(Path(r) for r in self.racines_recherche)
        racine_depot = Path(__file__).resolve().parent.parent
        return (racine_depot, racine_depot.parent)

    @property
    def index_fichiers(self):
        if self._index is not None:
            return self._index

        index = {}
        fichiers_vus = set()
        dossiers_vus = set()

        for racine in self._racines_recherche():
            if not racine.exists():
                continue
            for dossier, sous_dossiers, fichiers in os.walk(racine, followlinks=True):
                reel = os.path.realpath(dossier)
                if reel in dossiers_vus:
                    sous_dossiers[:] = []
                    continue
                dossiers_vus.add(reel)

                sous_dossiers[:] = [
                    d for d in sous_dossiers if d not in DOSSIERS_IGNORES
                ]

                for nom in fichiers:
                    chemin = Path(dossier) / nom
                    if chemin in fichiers_vus:
                        continue
                    fichiers_vus.add(chemin)
                    index.setdefault(nom, []).append(chemin)

        self._index = index
        return index

    def _departager(self, candidats, role, motif):
        non_recuperes = [p for p in candidats if p.is_symlink() and not p.exists()]
        candidats = [p for p in candidats if p.exists()]

        if not candidats:
            if non_recuperes:
                raise FileNotFoundError(
                    f"{role} : contenu non récupéré ({len(non_recuperes)} pointeur(s) annex)."
                )
            return None

        if len(candidats) > 1:
            racine_depot = self._racines_recherche()[0]
            dans_depot = [p for p in candidats if p.is_relative_to(racine_depot)]
            if len(dans_depot) == 1:
                return dans_depot[0]
            raise ValueError(f"{role} : ambiguïté, plusieurs fichiers correspondent.")
        return candidats[0]

    def _trouver_fichier(self, nom, role, obligatoire=True):
        chemin = self._departager(self.index_fichiers.get(nom, []), role, nom)
        if chemin is None and obligatoire:
            raise FileNotFoundError(f"{role} : fichier manquant.")
        return chemin

    def _trouver_dossier(self, motif, role, obligatoire=True):
        candidats = [
            chemin
            for nom, chemins in self.index_fichiers.items()
            if fnmatch(nom, motif)
            for chemin in chemins
        ]
        dossiers = {c.parent for c in candidats if c.exists()}

        if not dossiers:
            if obligatoire:
                raise FileNotFoundError(f"{role} : aucun fichier correspondant.")
            return None

        if len(dossiers) > 1:
            racine_depot = self._racines_recherche()[0]
            dans_depot = [d for d in dossiers if d.is_relative_to(racine_depot)]
            if len(dans_depot) == 1:
                return dans_depot[0]
            raise ValueError(f"{role} : ambiguïté sur les dossiers.")
        return dossiers.pop()

    def get_path_file(self, taches=None):
        taches = self.taches if taches is None else taches
        racines = self._racines_recherche()

        if taches == "things":
            chemin_tribe = self._trouver_fichier(
                f"{self.subject}.h5", f"embeddings {taches}"
            )
            dossier_embeddings = chemin_tribe.parent
        else:
            chemin_tribe = None
            dossier_embeddings = self._trouver_dossier(
                f"{taches}_*.h5", f"embeddings {taches}"
            )

        if self.flag_precision_voxel:
            nom_cneuromod = (
                f"{self.subject}_task-{taches}_space-T1w_desc-voxelwise_timeseries.h5"
            )
            nom_atlas = f"{self.subject}_task-{taches}_space-T1w_label-GMfromFS_desc-indivFunc_mask.nii.gz"
        else:
            nom_cneuromod = f"{self.subject}_task-{taches}_space-MNI152NLin2009cAsym_atlas-cneuromod26_desc-1134Parcels_timeseries.h5"
            nom_atlas = f"{self.subject}_task-{taches}_space-MNI152NLin2009cAsym_atlas-cneuromod26_desc-1134Parcels_dseg.nii.gz"

        chemin_cneuromod = self._trouver_fichier(nom_cneuromod, f"timeseries {taches}")
        chemin_atlas = self._trouver_fichier(nom_atlas, f"atlas {taches}")

        chemin_ROImask = self._trouver_fichier(
            f"{self.subject}_space-T1w_desc-ROImasks_voxelAnnotations.h5",
            f"ROImask {self.subject}",
            obligatoire=self.flag_precision_voxel,
        )
        chemin_annotations_parcelles = self._trouver_fichier(
            NOM_FICHIER_ANNOTATIONS_PARCELLES,
            "annotations des parcelles",
            obligatoire=not self.flag_precision_voxel,
        )
        chemin_anatomie = self._trouver_fichier(
            f"{self.subject}_desc-preproc_T1w.nii.gz",
            f"anatomie {self.subject}",
            obligatoire=False,
        )

        return CheminsProjet(
            root_encoding=Path(__file__).resolve().parent.parent,
            racines_recherche=racines,
            taches=taches,
            dossier_embeddings=dossier_embeddings,
            chemin_tribe=chemin_tribe,
            chemin_cneuromod=chemin_cneuromod,
            chemin_atlas=chemin_atlas,
            chemin_ROImask=chemin_ROImask,
            videos_par_nom=self._indexer_videos(taches),
            chemin_anatomie=chemin_anatomie,
            chemin_annotations_parcelles=chemin_annotations_parcelles,
        )

    def _indexer_videos(self, taches):
        if taches == "things":
            motif = f"{self.subject}_ses-*_task-*_run-*.mp4"
        else:
            motif = f"{taches}_*.mkv"

        videos = {}
        for nom, chemins in self.index_fichiers.items():
            if not fnmatch(nom, motif):
                continue
            existants = [c for c in chemins if c.exists()]
            if existants:
                videos[nom] = existants[0]
        return videos
