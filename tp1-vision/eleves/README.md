# TP1 – Vision par ordinateur

```
tp1_vision/
├── ma_vision.py       VOTRE module : le notebook y exporte vos fonctions validées
├── camera_utils.py    lecture de la caméra D405 (fourni)
├── captures/          images (exemple.png + profondeur + intrinsèques) et vos propres photos
├── notebook/          le TP guidé : tp1_vision.py  (+ exercices.py, checks.py ; vos réponses : mes_reponses.json)
└── live/              l'application : vision_node.py  + les tests ZeroMQ (zmq_test_pub.py, zmq_test_sub.py)
```

## Installation et lancement

```bash
pip install -r requirements.txt
marimo edit notebook/tp1_vision.py        # (ou : marimo run notebook/tp1_vision.py)
```

Toutes les commandes se lancent **depuis le dossier `tp1_vision/`**.

- Image par défaut : `captures/exemple.png` (+ profondeur `exemple_depth.npy`, intrinsèques `exemple.json`).
- Section « Source » : saisir un chemin, parcourir les fichiers, ou capturer depuis la D405 (enregistrée dans `captures/`).
- Votre travail est sauvegardé automatiquement dans `notebook/mes_reponses.json` : image, pixel, zone, réglages et code de vos fonctions. Au redémarrage, vous retrouvez tout.

## Le notebook

- **Petits exercices (✍️ À coder)** : quelques lignes à écrire sur les variables du notebook. Les tests se lancent quand on clique en dehors de l'éditeur ; les résultats restent verrouillés (🔒) jusqu'à validation.
- **Fonctions** : à la fin de chaque section, vous regroupez vos lignes dans une fonction ; le bouton **Exporter** (fin du notebook) écrit les fonctions validées dans `ma_vision.py`.
- Si une réponse n'est pas validée, le notebook utilise le résultat de la référence pour la suite : rien ne bloque.

| Section | Petits exercices | Fonction exportée |
|---|---|---|
| 1 Image | 1.1 lire un pixel · 1.2 BGR→RGB | |
| 2 Espaces de couleur | 2.1 BGR→HSV · 2.2 séparer H, S, V | |
| 3 Seuillage | 3.2a `inRange` · 3.2b isoler l'objet · 3.3a ouverture · 3.3b fermeture | `hsv_mask`, `clean_mask` |
| 4 Contour | 4.1 `findContours` · 4.2 tous les objets (aire ≥ seuil, triés) | `find_objects` |
| 5 Centroïde | 5.1 moments · 5.2 centroïde | `centroid(contour)` |
| 6 Profondeur | 6.2 pixel · 6.3a masque de l'objet · 6.3b médiane · 6.4 rétro-projection | `object_depth`, `deproject` |
| 7 ArUco | 7.2 détection · 7.3a coins 3D · 7.3b `solvePnP` | `detect_aruco`, `marker_pose` |

Revérifier votre module depuis un terminal : `python notebook/checks.py`.

## L'application (`live/`)

```bash
python live/vision_node.py --replay        # sans caméra : rejoue captures/exemple.* (fenêtre ouverte)
python live/vision_node.py                 # avec la D405
python live/vision_node.py --no-show       # sans fenêtre (robot, SSH)
python live/zmq_test_pub.py                # étape 0 : test ZeroMQ (un terminal)
python live/zmq_test_sub.py                # étape 0 : à compléter (autre terminal)
```

`process_frame` est le dernier exercice du notebook (section 8) : une fois exportée dans `ma_vision.py`, le nœud l'appelle à chaque image. Rien à compléter dans `vision_node.py` ; sa boucle est courte, avec une zone « 🔧 à vous » pour filtrer ou trier les objets avant publication. Par défaut, la fenêtre a des curseurs qui règlent le HSV, l'ouverture, la fermeture et l'aire minimale ; à la fermeture (`q`), les valeurs sont affichées pour les recopier en haut de `vision_node.py`.
Le nœud publie, sur le topic `vision/perception`, un message `{"timestamp", "frame_count", "objects": [{"id", "centroid", "area", "depth_m", "xyz_m"}]}`.
