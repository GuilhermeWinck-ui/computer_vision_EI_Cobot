"""
ma_vision.py — Votre module de vision
=====================================
Les fonctions encadrées par `# >>> exercice: ...` sont écrites et validées dans le
notebook, puis exportées ici par le bouton « Exporter ». Ce qui est en dehors des
repères est à vous : le notebook n'y touche jamais.

Vérifier ce fichier depuis un terminal :   python checks.py
"""

import cv2
import numpy as np

# >>> exercice: hsv_mask
def hsv_mask(img_bgr, lo, hi):
    """
    Masque binaire des pixels dont la couleur HSV est comprise entre lo et hi.

    Args:
        img_bgr : image BGR uint8 de forme (H, W, 3)  (ordre des canaux d'OpenCV)
        lo, hi  : bornes basse et haute (H, S, V), INCLUSES.  H ∈ [0, 179], S et V ∈ [0, 255]

    Returns:
        masque uint8 de forme (H, W), valant 255 là où le pixel est dans [lo, hi], 0 ailleurs.

    Fonctions utiles : cv2.cvtColor, cv2.inRange
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, np.array(lo, dtype=np.uint8), np.array(hi, dtype=np.uint8))
# <<< exercice: hsv_mask


# >>> exercice: clean_mask
def clean_mask(mask, k_open=7, k_close=15):
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
    ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open))
    kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
    out = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ko)
    return cv2.morphologyEx(out, cv2.MORPH_CLOSE, kc)
# <<< exercice: clean_mask


# >>> exercice: find_objects
def find_objects(mask, min_area=500):
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
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    objs = [c for c in contours if cv2.contourArea(c) >= min_area]
    return sorted(objs, key=cv2.contourArea, reverse=True)
# <<< exercice: find_objects


# >>> exercice: centroid
def centroid(contour):
    """
    Centroïde d'un objet : la moyenne des positions de ses pixels.

    Args:
        contour : contour de l'objet, tableau (N, 1, 2) de points (u, v)

    Returns:
        le couple (u_c, v_c) de nombres décimaux : u_c = colonne, v_c = ligne ;
        None si le contour n'a pas d'aire.

    Fonctions utiles : cv2.moments
    """
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None
    return (M["m10"] / M["m00"], M["m01"] / M["m00"])
# <<< exercice: centroid


# >>> exercice: object_depth
def object_depth(depth, contour, depth_scale):
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
    mask = np.zeros(depth.shape[:2], np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)
    valid = (mask > 0) & (depth > 0)
    if not valid.any():
        return None
    return float(np.median(depth[valid])) * depth_scale
# <<< exercice: object_depth


# >>> exercice: deproject
def deproject(u, v, z, K):
    """
    Rétro-projection : point 3D d'un pixel dont on connaît la profondeur.

    Args:
        u, v : coordonnées du pixel (u = colonne, v = ligne)
        z    : profondeur en mètres
        K    : matrice intrinsèque 3 × 3  [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]

    Returns:
        tableau NumPy [X, Y, Z] en mètres, dans le repère de la caméra.
    """
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    return np.array([z * (u - cx) / fx, z * (v - cy) / fy, z])
# <<< exercice: deproject


# >>> exercice: detect_aruco
def detect_aruco(img_bgr, dict_id=cv2.aruco.DICT_4X4_50):
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
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(dict_id), cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return []
    return [(int(i), c.reshape(4, 2)) for i, c in zip(ids.ravel(), corners)]
# <<< exercice: detect_aruco


# >>> exercice: marker_pose
def marker_pose(corners, L, K, dist=None):
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
    obj = np.array([[-L / 2, L / 2, 0], [L / 2, L / 2, 0], [L / 2, -L / 2, 0], [-L / 2, -L / 2, 0]], np.float32)
    ok, rvec, tvec = cv2.solvePnP(obj, np.asarray(corners, np.float32).reshape(4, 2), K, dist,
                                  flags=cv2.SOLVEPNP_IPPE_SQUARE)
    return (rvec, tvec) if ok else None
# <<< exercice: marker_pose


# >>> exercice: process_frame
def process_frame(color, depth, K, depth_scale, hsv_range, min_area=500, k_open=7, k_close=15):
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
    lo = (hsv_range["h_min"], hsv_range["s_min"], hsv_range["v_min"])
    hi = (hsv_range["h_max"], hsv_range["s_max"], hsv_range["v_max"])
    mask = clean_mask(hsv_mask(color, lo, hi), k_open, k_close)
    objects = []
    for c in find_objects(mask, min_area):
        ctr = centroid(c)
        z = object_depth(depth, c, depth_scale)
        if ctr is None or z is None:
            continue
        X, Y, Z = deproject(ctr[0], ctr[1], z, K)
        objects.append({"contour": c, "centroid": ctr, "area": float(cv2.contourArea(c)),
                        "depth_m": z, "xyz_m": (float(X), float(Y), float(Z))})
    return objects
# <<< exercice: process_frame
