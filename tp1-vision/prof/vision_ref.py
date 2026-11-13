"""
vision_ref.py — Solutions de référence (enseignant)
====================================================
Ce fichier sert de FILET DE SÉCURITÉ : si une fonction de votre module
`ma_vision.py` manque ou ne passe pas ses tests, le notebook et `vision_node.py`
utilisent la version de référence ci-dessous.

Ne le lisez qu'après avoir essayé : écrire la fonction soi-même est le but du TP.
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Réponse fonction globale
# ---------------------------------------------------------------------------
def hsv_mask(img_bgr, lo, hi):
    """Masque (0/255, uint8, (H, W)) des pixels dont (H, S, V) est dans [lo, hi] (bornes incluses)."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, np.array(lo, dtype=np.uint8), np.array(hi, dtype=np.uint8))


def clean_mask(mask, k_open=7, k_close=15):
    """Ouverture (supprime le bruit) puis fermeture (bouche les trous), motif elliptique."""
    ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open))
    kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
    out = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ko)
    return cv2.morphologyEx(out, cv2.MORPH_CLOSE, kc)

def find_objects(mask, min_area=500):
    """Tous les objets du masque : contours extérieurs d'aire >= min_area, du plus grand au plus petit (liste vide s'il n'y en a pas)."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    objs = [c for c in contours if cv2.contourArea(c) >= min_area]
    return sorted(objs, key=cv2.contourArea, reverse=True)


def centroid(contour):
    """Centroïde (u_c, v_c) d'un contour, ou None si son aire est nulle (m00 = 0)."""
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None
    return (M["m10"] / M["m00"], M["m01"] / M["m00"])


def object_depth(depth, contour, depth_scale):
    """Profondeur d'un objet en mètres : médiane des pixels de l'intérieur du contour dont la mesure est valide (≠ 0), ou None."""
    mask = np.zeros(depth.shape[:2], np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)
    valid = (mask > 0) & (depth > 0)
    if not valid.any():
        return None
    return float(np.median(depth[valid])) * depth_scale


def deproject(u, v, z, K):
    """Point 3D (X, Y, Z) en mètres, dans le repère de la caméra, du pixel (u, v) à la profondeur z."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    return np.array([z * (u - cx) / fx, z * (v - cy) / fy, z])


def detect_aruco(img_bgr, dict_id=cv2.aruco.DICT_4X4_50):
    """Marqueurs ArUco de l'image : liste de (id, coins) ; coins = tableau (4, 2) dans l'ordre haut-gauche, haut-droit, bas-droit, bas-gauche."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(dict_id), cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return []
    return [(int(i), c.reshape(4, 2)) for i, c in zip(ids.ravel(), corners)]


def marker_pose(corners, L, K, dist=None):
    """Pose du marqueur dans le repère caméra : (rvec, tvec), ou None si le calcul échoue."""
    obj = np.array([[-L / 2, L / 2, 0], [L / 2, L / 2, 0], [L / 2, -L / 2, 0], [-L / 2, -L / 2, 0]], np.float32)
    ok, rvec, tvec = cv2.solvePnP(obj, np.asarray(corners, np.float32).reshape(4, 2), K, dist,
                                  flags=cv2.SOLVEPNP_IPPE_SQUARE)
    return (rvec, tvec) if ok else None

# ---------------------------------------------------------------------------
# Réponses de référence des petits exercices du notebook (une entrée par exercice)
# Chaque fonction reçoit les entrées nommées de l'exercice et renvoie {variable: valeur}.
# ---------------------------------------------------------------------------

def _ref_pixel(img, pu, pv):
    b, g, r = (int(x) for x in img[pv, pu])
    return {"b": b, "g": g, "r": r}


def _ref_rgb(img):
    return {"rgb": cv2.cvtColor(img, cv2.COLOR_BGR2RGB)}


def _ref_hsv(img):
    return {"hsv": cv2.cvtColor(img, cv2.COLOR_BGR2HSV)}


def _ref_split(hsv):
    H, S, V = cv2.split(hsv)
    return {"H": H, "S": S, "V": V}


def _ref_mask(hsv, lo, hi):
    return {"mask": cv2.inRange(hsv, np.asarray(lo), np.asarray(hi))}


def _ref_objet(img, mask):
    return {"objet": cv2.bitwise_and(img, img, mask=mask)}


def _ref_open(mask, k_open):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(k_open), int(k_open)))
    return {"opened": cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)}


def _ref_close(opened, k_close):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(k_close), int(k_close)))
    return {"closed": cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k)}
    

def _ref_contours(mask_clean):
    c, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return {"contours": c}


def _ref_objects(contours, min_area):
    objs = [c for c in contours if cv2.contourArea(c) >= min_area]
    return {"objects": sorted(objs, key=cv2.contourArea, reverse=True)}


def _ref_moments(c_obj):
    return {"M": cv2.moments(c_obj)}


def _ref_centre(M):
    if M["m00"] == 0:
        return {"u_c": None, "v_c": None}
    return {"u_c": M["m10"] / M["m00"], "v_c": M["m01"] / M["m00"]}


def _ref_objmask(c_obj, mask_clean):
    obj_mask = np.zeros(mask_clean.shape[:2], np.uint8)
    cv2.drawContours(obj_mask, [c_obj], -1, 255, cv2.FILLED)
    return {"obj_mask": obj_mask}


def _ref_zpx(depth, depth_scale, pu, pv):
    return {"z_px": float(depth[pv, pu]) * depth_scale}


def _ref_zobj(depth, obj_mask, depth_scale):
    valid = (obj_mask > 0) & (depth > 0)
    return {"z_obj": float(np.median(depth[valid])) * depth_scale if valid.any() else None}


def _ref_xyz(u_c, v_c, z_obj, K):
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    return {"X": float(z_obj * (u_c - cx) / fx), "Y": float(z_obj * (v_c - cy) / fy), "Z": float(z_obj)}


def _ref_aruco(img, aruco_dict, params):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = cv2.aruco.ArucoDetector(aruco_dict, params).detectMarkers(gray)
    return {"gray": gray, "corners": corners, "ids": ids, "rejected": rejected}


def _ref_obj3d(marker_len):
    L = float(marker_len)
    return {"obj": np.array([[-L / 2, L / 2, 0], [L / 2, L / 2, 0], [L / 2, -L / 2, 0], [-L / 2, -L / 2, 0]], np.float32)}


def _ref_pnp(obj, img_pts, K, dist_coeffs):
    ok, rvec, tvec = cv2.solvePnP(obj, img_pts, K, dist_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    return {"ok": ok, "rvec": rvec, "tvec": tvec}


SNIPPET_REF = {
    "pixel": _ref_pixel, "rgb": _ref_rgb, "hsv": _ref_hsv, "split": _ref_split,
    "mask": _ref_mask, "objet": _ref_objet, "open": _ref_open, "close": _ref_close,
    "contours": _ref_contours, "objects": _ref_objects, "moments": _ref_moments, "centre": _ref_centre,
    "zpx": _ref_zpx, "objmask": _ref_objmask, "zobj": _ref_zobj, "xyz": _ref_xyz,
    "aruco": _ref_aruco, "obj3d": _ref_obj3d, "pnp": _ref_pnp,
}


def process_frame(color, depth, K, depth_scale, hsv_range, min_area=500, k_open=7, k_close=15):
    """Chaîne complète (solution de référence) : une entrée par objet gardé, du plus grand au plus petit.

    Chaque entrée : {"contour", "centroid": (u, v), "area", "depth_m", "xyz_m": (X, Y, Z)}.
    Les objets sans centroïde ou sans profondeur valide sont ignorés.
    """
    lo = (hsv_range["h_min"], hsv_range["s_min"], hsv_range["v_min"])
    hi = (hsv_range["h_max"], hsv_range["s_max"], hsv_range["v_max"])
    mask = clean_mask(hsv_mask(color, lo, hi), k_open, k_close)
    out = []
    for c in find_objects(mask, min_area):
        ctr = centroid(c)
        z = object_depth(depth, c, depth_scale)
        if ctr is None or z is None:
            continue
        out.append({"contour": c, "centroid": ctr, "area": float(cv2.contourArea(c)),
                    "depth_m": z, "xyz_m": tuple(float(x) for x in deproject(ctr[0], ctr[1], z, K))})
    return out
