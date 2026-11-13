"""
exercices.py — Énoncés des exercices et génération de `ma_vision.py`
=====================================================================
Fichier de l'enseignant : il décrit chaque exercice (énoncé de départ, liens vers
la documentation) et sait écrire / mettre à jour `ma_vision.py`.

`ma_vision.py` est votre module : le notebook y exporte vos fonctions validées,
et `vision_node.py` l'importe. Chaque fonction exportée est encadrée par deux
lignes repères (`# >>> exercice: nom` ... `# <<< exercice: nom`) : l'export ne
touche à rien en dehors de ces repères, vous pouvez donc écrire le reste
(par exemple `process_frame`) dans votre éditeur sans risque.
"""

import re

DOC = {
    "cvtcolor": ("cv2.cvtColor : référence", "https://docs.opencv.org/4.13.0/d8/d01/group__imgproc__color__conversions.html#gaf86c09fe702ed037c03c2bc603ceab14"),
    "inrange": ("cv2.inRange : référence", "https://docs.opencv.org/4.13.0/d2/de8/group__core__array.html#ga48af0ab51e36436c5d04340e036ce981"),
    "core_array": ("cv2.bitwise_and : référence", "https://docs.opencv.org/4.13.0/d2/de8/group__core__array.html#ga60b4d04b251ba5eb1392c34425497e14"),
    "imgproc_filter": ("Filtrage d'images (cherchez `morphologyEx`, `getStructuringElement`)", "https://docs.opencv.org/4.13.0/d4/d86/group__imgproc__filter.html"),
    "numpy_idx": ("NumPy : indexer un tableau", "https://numpy.org/doc/stable/user/basics.indexing.html"),
    "morph": ("Tutoriel : transformations morphologiques", "https://docs.opencv.org/4.13.0/d9/d61/tutorial_py_morphological_ops.html"),
    "morph2": ("Tutoriel : ouverture, fermeture et motifs", "https://docs.opencv.org/4.13.0/d3/dbe/tutorial_opening_closing_hats.html"),
    "contours_tuto": ("Tutoriel : contours, premiers pas (`findContours`)", "https://docs.opencv.org/3.4.20/d4/d73/tutorial_py_contours_begin.html"),
    "retr_modes": ("Modes de recherche des contours (`RetrievalModes`)", "https://docs.opencv.org/4.13.0/d3/dc0/group__imgproc__shape.html#ga819779b9857cc2f8601e6526a3a5bc71"),
    "approx_modes": ("Méthodes d'approximation (`ContourApproximationModes`)", "https://docs.opencv.org/4.13.0/d3/dc0/group__imgproc__shape.html#ga4303f45752694956374734a03c54d5ff"),
    "contour_feat": ("Tutoriel : caractéristiques d'un contour (aire, moments)", "https://docs.opencv.org/3.4.20/dd/d49/tutorial_py_contour_features.html"),
    "py_sorting": ("Python : trier avec `sorted` et `key`", "https://docs.python.org/3/howto/sorting.html"),
    "draw": ("Dessin (cherchez `drawContours`)", "https://docs.opencv.org/4.13.0/d6/d6e/group__imgproc__draw.html"),
    "np_zeros_like": ("NumPy : zeros_like", "https://numpy.org/doc/stable/reference/generated/numpy.zeros_like.html"),
    "np_median": ("NumPy : médiane", "https://numpy.org/doc/stable/reference/generated/numpy.median.html"),
    "calib3d": ("Modèle de caméra et formules de projection (module calib3d)", "https://docs.opencv.org/4.13.0/d9/d0c/group__calib3d.html"),
    "pnp": ("Calcul de pose : `solvePnP` et ses `flags`", "https://docs.opencv.org/4.13.0/d5/d1f/calib3d_solvePnP.html"),
    "aruco_tuto": ("Tutoriel : détection de marqueurs ArUco", "https://docs.opencv.org/4.13.0/d5/dae/tutorial_aruco_detection.html"),
    "py_dict": ("Python : les dictionnaires", "https://docs.python.org/3/tutorial/datastructures.html#dictionaries"),
    "aruco_det": ("Classe `ArucoDetector` (méthode `detectMarkers`)", "https://docs.opencv.org/4.13.0/d2/d1a/classcv_1_1aruco_1_1ArucoDetector.html"),
}

EXERCICES = [
    {
        "name": "hsv_mask",
        "section": "3.4",
        "title": "Écrire la fonction `hsv_mask`",
        "intro": ("Vous venez d'écrire ces lignes avec les variables du notebook (`hsv`, `lo`, `hi`…). Regroupez-les dans une **fonction réutilisable** : "
                  "tout ce dont elle a besoin lui est passé en **paramètres**. C'est cette fonction que `vision_node.py` appellera à chaque image."),
        "steps": ["reprenez vos lignes 2.1 et 3.2a", "remplacez les variables du notebook par les paramètres", "n'oubliez pas le `return`"],
        "docs": ["cvtcolor", "inrange"],
        "stub": '''def hsv_mask(img_bgr, lo, hi):
    """
    Masque binaire des pixels dont la couleur HSV est comprise entre lo et hi.

    Args:
        img_bgr : image BGR uint8 de forme (H, W, 3)  (ordre des canaux d'OpenCV)
        lo, hi  : bornes basse et haute (H, S, V), INCLUSES.  H ∈ [0, 179], S et V ∈ [0, 255]

    Returns:
        masque uint8 de forme (H, W), valant 255 là où le pixel est dans [lo, hi], 0 ailleurs.

    Fonctions utiles : cv2.cvtColor, cv2.inRange
    """
    # votre code ici
    raise NotImplementedError
''',
    },
    {
        "name": "clean_mask",
        "section": "3.4",
        "title": "Écrire la fonction `clean_mask`",
        "intro": ("Même démarche : regroupez votre ouverture et votre fermeture dans une fonction qui prend le masque et les deux tailles de motif en paramètres."),
        "steps": ["reprenez vos lignes 3.3a et 3.3b", "remplacez `k_open` et `k_close` du notebook par les paramètres", "n'oubliez pas le `return`"],
        "docs": ["imgproc_filter", "morph", "morph2"],
        "stub": '''def clean_mask(mask, k_open=7, k_close=15):
    """
    Nettoie un masque binaire : ouverture (supprime le bruit) puis fermeture (bouche les trous).

    Args:
        mask    : masque uint8 (H, W) valant 0 ou 255
        k_open  : côté du motif d'ouverture, en pixels (impair ; 1 = aucune ouverture)
        k_close : côté du motif de fermeture, en pixels (impair ; 1 = aucune fermeture)

    Returns:
        nouveau masque uint8 (H, W). Motifs ELLIPTIQUES ; ne modifie pas `mask`.

    Fonctions utiles : cv2.getStructuringElement, cv2.morphologyEx, cv2.MORPH_OPEN, cv2.MORPH_CLOSE
    """
    # votre code ici
    raise NotImplementedError
''',
    },
    {
        "name": "find_objects",
        "section": "4.3",
        "title": "Écrire la fonction `find_objects`",
        "intro": "Regroupez 4.1 et 4.2 dans une fonction : elle reçoit le **masque** et le **seuil d'aire** en paramètres et renvoie **tous les objets** : la liste des contours d'aire suffisante, du plus grand au plus petit.",
        "steps": ['reprenez vos lignes 4.1 et 4.2', '`mask_clean` et `min_area` deviennent des paramètres', 'aucun objet : renvoyez une liste vide `[]`'],
        "docs": ['contours_tuto', 'retr_modes', 'approx_modes', 'contour_feat'],
        "stub": '''def find_objects(mask, min_area=500):
    """
    Tous les objets d'un masque binaire : ses contours extérieurs dont l'aire atteint min_area.

    Args:
        mask     : masque uint8 (H, W) valant 0 ou 255
        min_area : aire minimale (en pixels²) pour qu'un contour compte comme un objet

    Returns:
        liste de contours (chacun : tableau (N, 1, 2) de points (u, v)), du plus grand au plus
        petit en aire. Liste vide s'il n'y a aucun objet.

    Fonctions utiles : cv2.findContours, cv2.contourArea, sorted
    """
    # votre code ici
    raise NotImplementedError
''',
    },
    {
        "name": "centroid",
        "section": "5.3",
        "title": "Écrire la fonction `centroid`",
        "intro": "Regroupez 5.1 et 5.2 : la fonction reçoit **le contour d'un objet** et renvoie son centroïde `(u_c, v_c)`, ou `None` si le contour n'a pas d'aire. Dans l'application, vous l'appellerez pour **chaque objet**.",
        "steps": ['reprenez vos lignes 5.1 et 5.2', '`c_obj` devient le paramètre', 'renvoyez le couple `(u_c, v_c)` ; aire nulle → `None`'],
        "docs": ['contour_feat'],
        "stub": '''def centroid(contour):
    """
    Centroïde d'un objet : la moyenne des positions de ses pixels.

    Args:
        contour : contour de l'objet, tableau (N, 1, 2) de points (u, v)

    Returns:
        le couple (u_c, v_c) de nombres décimaux : u_c = colonne, v_c = ligne ;
        None si le contour n'a pas d'aire.

    Fonctions utiles : cv2.moments
    """
    # votre code ici
    raise NotImplementedError
''',
    },
    {
        "name": "object_depth",
        "section": "6.5",
        "title": "Écrire la fonction `object_depth`",
        "intro": "Regroupez 6.3a et 6.3b : la fonction reçoit la carte de profondeur, **le contour de l'objet** et `depth_scale`, et renvoie la profondeur de l'objet **en mètres**.",
        "steps": ["reprenez vos lignes 6.3a (masque de l'objet) et 6.3b (médiane)", '`depth`, `c_obj` et `depth_scale` deviennent des paramètres', 'aucun pixel valide : renvoyez `None`'],
        "docs": ['draw', 'np_median', 'numpy_idx'],
        "stub": '''def object_depth(depth, contour, depth_scale):
    """
    Profondeur d'un objet, en mètres.

    Args:
        depth       : carte de profondeur uint16 (H, W), en unités brutes ; 0 = pas de mesure
        contour     : contour de l'objet, tableau (N, 1, 2) de points (u, v)
        depth_scale : mètres par unité brute

    Returns:
        la profondeur (float, en mètres) ; None si aucun pixel de l'objet n'a de mesure valide.

    Fonctions utiles : cv2.drawContours, np.median
    """
    # votre code ici
    raise NotImplementedError
''',
    },
    {
        "name": "deproject",
        "section": "6.5",
        "title": "Écrire la fonction `deproject`",
        "intro": "Regroupez 6.4 : la fonction reçoit un pixel, sa profondeur et la matrice `K`, et renvoie le **point 3D** dans le repère de la caméra.",
        "steps": ["reprenez vos lignes 6.4", "`fx, fy, cx, cy` se lisent dans `K` (passée en paramètre)", "renvoyez un tableau NumPy `[X, Y, Z]`"],
        "docs": ["calib3d"],
        "stub": '''def deproject(u, v, z, K):
    """
    Rétro-projection : point 3D d'un pixel dont on connaît la profondeur.

    Args:
        u, v : coordonnées du pixel (u = colonne, v = ligne)
        z    : profondeur en mètres
        K    : matrice intrinsèque 3 × 3  [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]

    Returns:
        tableau NumPy [X, Y, Z] en mètres, dans le repère de la caméra.
    """
    # votre code ici
    raise NotImplementedError
''',
    },
    {
        "name": "detect_aruco",
        "section": "7.5",
        "title": "Écrire la fonction `detect_aruco`",
        "intro": "Regroupez 7.2 : la fonction reçoit l'image **couleur** et le dictionnaire, et renvoie la **liste des marqueurs** trouvés. Attention : quand rien n'est détecté, `ids` vaut `None`.",
        "steps": ["reprenez vos lignes 7.2 (gris, détecteur, détection)", "le dictionnaire se crée à partir de `dict_id`", "mettez en forme : liste de `(id, coins)` avec des coins `(4, 2)` ; aucun marqueur → liste vide"],
        "docs": ["aruco_tuto", "aruco_det"],
        "stub": '''def detect_aruco(img_bgr, dict_id=cv2.aruco.DICT_4X4_50):
    """
    Détecte les marqueurs ArUco d'une image.

    Args:
        img_bgr : image BGR uint8 (H, W, 3)
        dict_id : identifiant du dictionnaire, par exemple cv2.aruco.DICT_4X4_50

    Returns:
        liste de couples (id, coins) : id est un entier, coins un tableau (4, 2) de points (u, v)
        dans l'ordre haut-gauche, haut-droit, bas-droit, bas-gauche.
        Liste vide s'il n'y a aucun marqueur.

    Fonctions utiles : cv2.aruco.getPredefinedDictionary, cv2.aruco.ArucoDetector, detectMarkers
    """
    # votre code ici
    raise NotImplementedError
''',
    },
    {
        "name": "marker_pose",
        "section": "7.5",
        "title": "Écrire la fonction `marker_pose`",
        "intro": "Regroupez 7.3a et 7.3b : la fonction reçoit les coins détectés, la taille réelle `L` et les intrinsèques, et renvoie la **pose** du marqueur dans le repère de la caméra.",
        "steps": ["reprenez vos lignes 7.3a (coins 3D) et 7.3b (`solvePnP`)", "`marker_len` devient `L`, `K` et `dist` sont des paramètres", "renvoyez `(rvec, tvec)`"],
        "docs": ["pnp", "aruco_tuto"],
        "stub": '''def marker_pose(corners, L, K, dist=None):
    """
    Pose d'un marqueur carré dans le repère de la caméra.

    Args:
        corners : coins détectés, tableau (4, 2) en pixels, ordre haut-gauche, haut-droit, bas-droit, bas-gauche
        L       : côté réel du marqueur, en mètres
        K       : matrice intrinsèque 3 × 3
        dist    : coefficients de distorsion (ou None)

    Returns:
        (rvec, tvec) : rotation (vecteur de Rodrigues) et translation en mètres ;
        None si le calcul échoue.

    Fonctions utiles : cv2.solvePnP (flags=cv2.SOLVEPNP_IPPE_SQUARE)
    """
    # votre code ici
    raise NotImplementedError
''',
    },
    {
        "name": "process_frame",
        "section": "8",
        "title": "Écrire la fonction `process_frame`",
        "intro": ("Vous avez écrit toutes les briques : il reste à les **enchaîner**. `process_frame` reçoit une image couleur, la profondeur et les intrinsèques, et renvoie **la liste de tous les objets** "
                  "avec leur position 3D. C'est **cette fonction** que `vision_node.py` appelle à chaque image.\n\n"
                  "Dans votre code, appelez directement `hsv_mask`, `clean_mask`, `find_objects`, `centroid`, `object_depth` et `deproject` (celles que vous avez validées ; sinon la version de référence est utilisée)."),
        "steps": ["`hsv_range` est un dictionnaire : fabriquez `lo` et `hi` (H, S, V) à partir de ses clés",
                  "enchaînez `hsv_mask` → `clean_mask` (avec `k_open` et `k_close`) → `find_objects`",
                  "pour **chaque** contour : centroïde, profondeur, point 3D ; **ignorez** l'objet si le centroïde ou la profondeur vaut `None`",
                  "renvoyez la liste de dictionnaires (du plus grand objet au plus petit)"],
        "docs": ["py_dict"],
        "stub": '''def process_frame(color, depth, K, depth_scale, hsv_range, min_area=500, k_open=7, k_close=15):
    """
    Détecte tous les objets de la couleur demandée et calcule leur position 3D.

    Args:
        color       : image BGR (H, W, 3) uint8
        depth       : carte de profondeur (H, W) uint16, valeurs BRUTES (0 = pas de mesure)
        K           : matrice intrinsèque 3 × 3
        depth_scale : mètres par unité brute
        hsv_range   : dict avec les clés h_min, h_max, s_min, s_max, v_min, v_max
        min_area    : aire minimale d'un objet gardé (pixels²)
        k_open, k_close : tailles des motifs d'ouverture et de fermeture, transmises à `clean_mask`

    Returns:
        liste (du plus grand objet au plus petit) de dictionnaires :
            {"contour": contour, "centroid": (u, v), "area": aire en pixels²,
             "depth_m": profondeur en mètres, "xyz_m": (X, Y, Z) en mètres}
        Un objet dont le centroïde ou la profondeur est None est ignoré. Aucun objet : liste vide [].

    Fonctions utiles : hsv_mask, clean_mask, find_objects, centroid, object_depth, deproject, cv2.contourArea
    """
    # votre code ici
    raise NotImplementedError
''',
    },
]

BY_NAME = {e["name"]: e for e in EXERCICES}


# ---------------------------------------------------------------------------
# Petits exercices du notebook : quelques lignes écrites sur les VARIABLES du notebook
# ---------------------------------------------------------------------------

SNIPPETS = [
    {
        "id": "pixel", "section": "1.1", "title": "Lire un pixel",
        "intro": "Lisez les trois valeurs du pixel que vous avez choisi sur l'image.",
        "given": [("img", "image BGR, tableau (H, W, 3)"), ("pu", "colonne u du pixel"), ("pv", "ligne v du pixel")],
        "outputs": [("b, g, r", "les trois valeurs de ce pixel")],
        "expect": "trois nombres entre 0 et 255 : la couleur du pixel, dans l'ordre des canaux d'OpenCV.",
        "docs": ["numpy_idx"],
        "starter": "b, g, r = None, None, None\n",
    },
    {
        "id": "rgb", "section": "1.2", "title": "Remettre les canaux dans l'ordre pour l'affichage",
        "intro": "Les afficheurs attendent R, G, B, mais `img` est rangée en B, G, R. Fabriquez l'image prête à afficher.",
        "given": [("img", "image BGR, tableau (H, W, 3) uint8")],
        "outputs": [("rgb", "la même image, canaux dans l'ordre R, G, B")],
        "expect": "la même image que `img`, mais avec les canaux R et B échangés, pour que les couleurs s'affichent correctement.",
        "docs": ["cvtcolor"],
        "starter": "rgb = None\n",
    },
    {
        "id": "hsv", "section": "2.1", "title": "Passer l'image en HSV",
        "intro": "Convertissez l'image dans l'espace HSV : la couleur (teinte) y est séparée de la luminosité.",
        "given": [("img", "image BGR, tableau (H, W, 3) uint8")],
        "outputs": [("hsv", "la même image en HSV, tableau (H, W, 3) uint8")],
        "expect": 'un tableau de la même forme que `img`, dont les 3 valeurs par pixel sont maintenant (H, S, V).',
        "docs": ["cvtcolor"],
        "starter": "hsv = None\n",
    },
    {
        "id": "split", "section": "2.2", "title": "Séparer les trois canaux",
        "intro": "Récupérez la teinte, la saturation et la valeur sous forme de trois images (une par canal).",
        "given": [("hsv", "image HSV, tableau (H, W, 3)")],
        "outputs": [("H, S, V", "trois tableaux (H, W), un par canal")],
        "expect": 'trois images en niveaux de gris de forme (H, W) : la teinte, la saturation et la valeur de chaque pixel.',
        "docs": ["core_array"],
        "starter": "H, S, V = None, None, None\n",
    },
    {
        "id": "mask", "section": "3.2a", "title": "Seuiller : fabriquer le masque",
        "intro": "Gardez les pixels dont (H, S, V) est entre les bornes réglées en 3.1. Le masque vaut 255 là où le pixel est gardé, 0 ailleurs.",
        "given": [("hsv", "image HSV"), ("lo", "bornes basses [H, S, V]"), ("hi", "bornes hautes [H, S, V]")],
        "outputs": [("mask", "tableau (H, W) uint8 valant 0 ou 255")],
        "expect": 'une image noire et blanche : blanc (255) là où le pixel est dans les bornes, noir (0) ailleurs.',
        "docs": ["inrange"],
        "starter": "mask = None\n",
    },
    {
        "id": "objet", "section": "3.2b", "title": "Isoler l'objet dans l'image",
        "intro": "On veut voir **uniquement l'objet** : garder la couleur de `img` là où `mask` est blanc, et mettre tout le reste en **noir**. Indice : une opération `bitwise_…` d'OpenCV combine deux images pixel à pixel et accepte un paramètre `mask` qui limite le calcul aux pixels non nuls.",
        "given": [("img", "image BGR"), ("mask", "votre masque (0 ou 255)")],
        "outputs": [("objet", "image BGR : l'objet seul sur fond noir")],
        "expect": "l'image d'origine, en couleur, mais noire partout où le masque vaut 0.",
        "docs": ["core_array"],
        "starter": "objet = None\n",
    },
    {
        "id": "open", "section": "3.3a", "title": "Ouverture : supprimer le bruit",
        "intro": "Appliquez une **ouverture** au masque, avec un motif **elliptique** de côté `k_open`.",
        "given": [("mask", "votre masque brut"), ("k_open", "côté du motif, réglé par le curseur")],
        "outputs": [("opened", "le masque après ouverture")],
        "expect": "le masque sans les petits points blancs isolés, l'objet gardant sa taille.",
        "docs": ["morph", "morph2"],
        "starter": "opened = None\n",
    },
    {
        "id": "close", "section": "3.3b", "title": "Fermeture : boucher les trous",
        "intro": "Appliquez une **fermeture** au masque **déjà ouvert**, avec un motif elliptique de côté `k_close`.",
        "given": [("opened", "le masque après ouverture"), ("k_close", "côté du motif, réglé par le curseur")],
        "outputs": [("closed", "le masque propre")],
        "expect": "le masque sans les petits trous noirs dans l'objet.",
        "docs": ["morph", "morph2"],
        "starter": "closed = None\n",
    },
    {
        "id": 'contours',
        "section": '4.1',
        "title": 'Trouver les contours du masque',
        "intro": "Récupérez les contours des régions blanches de votre masque propre. Deux choix à faire en lisant la doc : le **mode de recherche** (voulez-vous tous les contours, ou seulement les **extérieurs** ?) et la **méthode d'approximation** (voulez-vous **tous** les points, ou seulement les **extrémités des segments** ?). Le TP demande : contours extérieurs, extrémités des segments.",
        "given": [('mask_clean', 'votre masque propre (0 ou 255)')],
        "outputs": [('contours', 'la liste des contours : chacun est un tableau `(N, 1, 2)` de points `(u, v)`')],
        "expect": "une liste de contours, un par région blanche **extérieure** (les trous et les objets emboîtés ne comptent pas). La fonction d'OpenCV renvoie **deux** valeurs : à vous de ne garder que la bonne.",
        "docs": ['contours_tuto', 'retr_modes', 'approx_modes'],
        "starter": 'contours = None\n',
    },
    {
        "id": 'objects',
        "section": '4.2',
        "title": 'Garder tous les objets',
        "intro": "Un contour trop petit est du bruit, pas un objet. Gardez **tous** les contours dont l'aire atteint le seuil `min_area`, **triés du plus grand au plus petit** : c'est la liste de vos objets. S'il n'y en a aucun, la liste est vide.",
        "given": [('contours', 'la liste de contours de 4.1'), ('min_area', 'aire minimale en pixels², réglée par le curseur')],
        "outputs": [('objects', "liste de contours `(N, 1, 2)`, du plus grand au plus petit ; `[]` s'il n'y en a aucun")],
        "expect": "la liste des contours d'aire ≥ `min_area`, dans l'ordre décroissant des aires (le n° 1 est le plus grand).",
        "docs": ['contour_feat', 'py_sorting'],
        "starter": 'objects = None\n',
    },
    {
        "id": 'moments',
        "section": '5.1',
        "title": "Calculer les moments d'un objet",
        "intro": "Les **moments** résument une région par quelques sommes. Calculez ceux du contour `c_obj` : `m00` est alors l'**aire** du contour.",
        "given": [('c_obj', "le contour de l'objet inspecté (tableau `(N, 1, 2)`), choisi en 4.2")],
        "outputs": [('M', 'le dictionnaire de moments (`m00`, `m10`, `m01`, …)')],
        "expect": 'un dictionnaire dont `M["m00"]` est l\'aire du contour.',
        "docs": ['contour_feat'],
        "starter": 'M = None\n',
    },
    {
        "id": 'centre',
        "section": '5.2',
        "title": 'En déduire le centroïde',
        "intro": "Le centroïde est la **moyenne des positions** des pixels : deux de ces moments, divisés par `m00`, donnent sa colonne `u_c` et sa ligne `v_c`. Si `m00 = 0` (contour réduit à un point), il n'y a pas de centroïde : laissez `None`.",
        "given": [('M', 'les moments de 5.1')],
        "outputs": [('u_c, v_c', 'colonne et ligne du centroïde (nombres décimaux)')],
        "expect": "deux nombres décimaux (sous-pixel) : la position moyenne des pixels de l'objet, `u_c` = colonne, `v_c` = ligne.",
        "docs": ['contour_feat'],
        "starter": 'u_c, v_c = None, None\n',
    },
    {
        "id": "zpx", "section": "6.2", "title": "Lire la profondeur d'un pixel",
        "intro": "Lisez la profondeur du pixel choisi en 1.1, **en mètres**. Rappel : `depth` contient des entiers bruts.",
        "given": [("depth", "carte de profondeur, tableau (H, W) uint16, alignée sur l'image"), ("depth_scale", "mètres par unité brute"),
                  ("pu", "colonne u du pixel"), ("pv", "ligne v du pixel")],
        "outputs": [("z_px", "la profondeur de ce pixel, en mètres")],
        "expect": "un nombre : la distance du pixel à la caméra en mètres (0 si le pixel n'a pas de mesure).",
        "docs": ["numpy_idx"],
        "starter": "z_px = None\n",
    },
    {
        "id": 'objmask',
        "section": '6.3a',
        "title": "Le masque de l'objet",
        "intro": "Pour lire la profondeur **de l'objet**, il faut savoir quels pixels le composent. Fabriquez son masque : une image noire de la même taille que l'image, avec **tout l'intérieur du contour** en blanc (255).\n\n**Pour dessiner** : `cv2.drawContours(image, contours, contourIdx, color, thickness)` dessine **sur** `image` (modifiée en place). `contours` est une **liste** de contours (donc `[c_obj]`, pas `c_obj`) ; `contourIdx = -1` dessine tous ceux de la liste ; `thickness = cv2.FILLED` remplit l'intérieur au lieu de tracer le trait. Pour l'image noire : `np.zeros_like`.",
        "given": [('c_obj', "le contour de l'objet inspecté"), ('mask_clean', "votre masque propre : sa forme (H, W) donne la taille de l'image")],
        "outputs": [('obj_mask', 'tableau `(H, W)` uint8 valant 0 ou 255')],
        "expect": "une image noire et blanche : blanc (255) sur **toute la surface** de l'objet (pas seulement son trait), noir ailleurs.",
        "docs": ['draw', 'np_zeros_like'],
        "starter": 'obj_mask = None\n',
    },
    {
        "id": 'zobj',
        "section": '6.3b',
        "title": "La profondeur de l'objet",
        "intro": "Un pixel isolé est fragile (il peut être invalide, ou au bord). Estimez la profondeur de **l'objet** à partir de **tous** ses pixels, en ignorant ceux sans mesure, avec un estimateur **robuste aux valeurs aberrantes**. Résultat en **mètres**.",
        "given": [('depth', 'carte de profondeur uint16 (H, W) ; 0 = pas de mesure'), ('obj_mask', "masque de l'objet, de 6.3a (0 ou 255)"), ('depth_scale', 'mètres par unité brute')],
        "outputs": [('z_obj', "la profondeur de l'objet, en mètres")],
        "expect": 'un nombre en mètres, calculé seulement sur les pixels qui sont **dans le masque ET mesurés**.',
        "docs": ['np_median', 'numpy_idx'],
        "starter": 'z_obj = None\n',
    },
    {
        "id": "xyz", "section": "6.4", "title": "Du pixel au point 3D",
        "intro": "Avec le centroïde `(u_c, v_c)`, la profondeur `z_obj` et les intrinsèques de `K`, retrouvez la position 3D de l'objet dans le repère de la caméra (modèle sténopé).",
        "given": [("u_c, v_c", "centroïde de l'objet (pixels)"), ("z_obj", "profondeur de l'objet (mètres)"),
                  ("K", "matrice intrinsèque 3 × 3 : `fx = K[0, 0]`, `fy = K[1, 1]`, `cx = K[0, 2]`, `cy = K[1, 2]`")],
        "outputs": [("X, Y, Z", "coordonnées du point en mètres, repère de la caméra")],
        "expect": "trois nombres en mètres : X vers la droite, Y vers le bas, Z = la profondeur mesurée.",
        "docs": ["calib3d"],
        "starter": "X, Y, Z = None, None, None\n",
    },
    {
        "id": "aruco", "section": "7.2", "title": "Détecter les marqueurs",
        "intro": "Convertissez l'image en **niveaux de gris**, puis lancez la détection ArUco avec le dictionnaire `aruco_dict` et les réglages `params`.",
        "given": [("img", "image BGR"), ("aruco_dict", "le dictionnaire choisi en 7.1"), ("params", "les réglages du détecteur (`DetectorParameters`)")],
        "outputs": [("gray", "l'image en niveaux de gris"), ("corners, ids, rejected", "les trois résultats de la détection")],
        "expect": "`corners` : un tableau `(1, 4, 2)` par marqueur ; `ids` : un tableau d'identifiants, ou `None` si rien n'est détecté ; `rejected` : les candidats refusés.",
        "docs": ["aruco_tuto", "aruco_det", "cvtcolor"],
        "starter": "gray = None\ncorners, ids, rejected = None, None, None\n",
    },
    {
        "id": "obj3d", "section": "7.3a", "title": "Les coins du marqueur dans son propre repère",
        "intro": "Pour calculer la pose, OpenCV a besoin des **4 coins du marqueur en 3D**, dans **son** repère : origine au **centre**, le marqueur dans le plan `z = 0`, côté `L`. Donnez-les **dans le même ordre que les coins détectés** : haut-gauche, haut-droit, bas-droit, bas-gauche (x vers la droite, y vers le haut).",
        "given": [("marker_len", "côté L du marqueur, en mètres")],
        "outputs": [("obj", "tableau `(4, 3)` de décimaux : un coin `(x, y, z)` par ligne")],
        "expect": "4 lignes `(x, y, 0)` : un carré de côté `L` centré sur l'origine.",
        "docs": ["pnp"],
        "starter": "obj = None\n",
    },
    {
        "id": "pnp", "section": "7.3b", "title": "Calculer la pose avec solvePnP",
        "intro": "On connaît les 4 coins en **3D** (`obj`) et leurs **4 projections** dans l'image (`img_pts`) : `solvePnP` retrouve la pose `[R | t]` du marqueur dans le repère de la caméra. Pour un marqueur carré, utilisez la variante prévue pour cela (`flags`).",
        "given": [("obj", "coins 3D, tableau (4, 3)"), ("img_pts", "coins détectés, tableau (4, 2) en pixels"), ("K", "matrice intrinsèque"), ("dist_coeffs", "distorsion")],
        "outputs": [("ok, rvec, tvec", "succès, rotation (vecteur de Rodrigues) et translation en mètres")],
        "expect": "`tvec` = position du **centre** du marqueur dans le repère caméra, en mètres (Z ≈ la distance).",
        "docs": ["pnp"],
        "starter": "ok, rvec, tvec = None, None, None\n",
    },
]

SNIP_BY_ID = {sp["id"]: sp for sp in SNIPPETS}

HEADER = '''"""
ma_vision.py — Votre module de vision
=====================================
Les fonctions encadrées par `# >>> exercice: ...` sont écrites et validées dans le
notebook, puis exportées ici par le bouton « Exporter ». Ce qui est en dehors des
repères est à vous : le notebook n'y touche jamais.

Vérifier ce fichier depuis un terminal (dossier du TP) :   python notebook/checks.py
"""

import cv2
import numpy as np

'''


def _block(name, code):
    return f"# >>> exercice: {name}\n{code.rstrip()}\n# <<< exercice: {name}\n"


def build_module(codes=None, existing=None):
    """
    Texte de `ma_vision.py`.
    codes    : {nom: code validé}. Les exercices absents gardent leur énoncé de départ (non écrit).
    existing : texte actuel du fichier (ou None). Seuls les blocs balisés sont remplacés.
    """
    codes = codes or {}
    text = existing if existing else HEADER + "\n\n".join(_block(e["name"], e["stub"]) for e in EXERCICES)
    for ex in EXERCICES:
        name = ex["name"]
        pat = re.compile(rf"# >>> exercice: {name}\n.*?# <<< exercice: {name}\n", re.S)
        if name in codes:
            new = _block(name, codes[name])
            if pat.search(text):
                text = pat.sub(lambda _m: new, text, count=1)
            else:
                text = text.rstrip() + "\n\n\n" + new
        elif not pat.search(text):                       # exercice nouveau dans un vieux fichier
            text = text.rstrip() + "\n\n\n" + _block(name, ex["stub"])
    return text


def read_block(text, name):
    """Code de l'exercice `name` dans le texte d'un ma_vision.py (None s'il n'y est pas)."""
    m = re.search(rf"# >>> exercice: {name}\n(.*?)# <<< exercice: {name}\n", text, re.S)
    return m.group(1).rstrip() + "\n" if m else None


def load_code(name, path):
    """Point de départ de l'éditeur : votre dernière version exportée, sinon l'énoncé."""
    try:
        with open(path, encoding="utf-8") as f:
            return read_block(f.read(), name) or BY_NAME[name]["stub"]
    except OSError:
        return BY_NAME[name]["stub"]
