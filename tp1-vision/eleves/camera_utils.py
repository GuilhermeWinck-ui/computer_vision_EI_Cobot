"""
camera_utils.py — Initialisation et lecture de la caméra Intel RealSense D405
===============================================================================
Fourni complet. Vous n'avez pas à modifier ce fichier.

La D405 est une caméra RGB-D : elle fournit simultanément :
  - une image couleur (BGR, 848×480 par défaut)
  - une image de profondeur (uint16), alignée pixel à pixel sur la couleur

Convention (la même que dans le notebook) :
  - profondeur en mètres = depth[v, u] * depth_scale   (depth_scale lu sur la caméra)
  - K = matrice intrinsèque 3×3 de la caméra couleur, à utiliser pour déprojeter.
"""

import numpy as np
import pyrealsense2 as rs


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

DEFAULT_WIDTH  = 848
DEFAULT_HEIGHT = 480
DEFAULT_FPS    = 30


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def init_d405(width: int = DEFAULT_WIDTH,
              height: int = DEFAULT_HEIGHT,
              fps: int = DEFAULT_FPS):
    """
    Initialise le pipeline RealSense D405 et retourne (pipeline, align).

    Returns:
        pipeline : objet rs.pipeline actif
        align    : objet rs.align pour aligner depth sur color

    Exemple :
        pipeline, align = init_d405()
        color, depth = get_aligned_frames(pipeline, align)
        K, dist      = get_intrinsics(pipeline)
        depth_scale  = get_depth_scale(pipeline)
    """
    pipeline = rs.pipeline()
    config   = rs.config()

    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16,  fps)

    pipeline.start(config)
    align = rs.align(rs.stream.color)

    print(f"[D405] Caméra initialisée : {width}×{height} @ {fps} fps")
    return pipeline, align


def stop_d405(pipeline):
    """Arrête proprement le pipeline."""
    pipeline.stop()
    print("[D405] Caméra arrêtée.")


# ---------------------------------------------------------------------------
# Lecture d'une frame
# ---------------------------------------------------------------------------

def get_aligned_frames(pipeline, align) -> tuple[np.ndarray, np.ndarray]:
    """
    Capture une paire (couleur, profondeur) alignée.

    Returns:
        color_image : ndarray uint8  (H, W, 3) BGR
        depth_image : ndarray uint16 (H, W) — valeurs BRUTES (pas des mm !).
                      Profondeur en mètres = depth_image * depth_scale
                      (voir get_depth_scale).

    Note :
        Une valeur depth_image[v, u] == 0 signifie que la mesure est invalide
        (hors portée, occlusion, surface réfléchissante, etc.).
    """
    frames        = pipeline.wait_for_frames()
    aligned       = align.process(frames)

    color_frame   = aligned.get_color_frame()
    depth_frame   = aligned.get_depth_frame()

    color_image   = np.asanyarray(color_frame.get_data())
    depth_image   = np.asanyarray(depth_frame.get_data())

    return color_image, depth_image


# ---------------------------------------------------------------------------
# Paramètres de la caméra
# ---------------------------------------------------------------------------

def get_depth_scale(pipeline) -> float:
    """
    Facteur d'échelle depth → mètres (lu sur le capteur de la caméra).
    À utiliser tel quel : ne supposez pas une valeur (0.001, 0.0001...).
    """
    sensor = pipeline.get_active_profile() \
                     .get_device() \
                     .first_depth_sensor()
    return sensor.get_depth_scale()


def get_intrinsics(pipeline) -> tuple[np.ndarray, np.ndarray]:
    """
    Intrinsèques de la caméra couleur (= celles de la profondeur alignée).

    Returns:
        K    : ndarray (3, 3) = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        dist : ndarray (5,)   coefficients de distorsion
    """
    intr = pipeline.get_active_profile() \
                   .get_stream(rs.stream.color) \
                   .as_video_stream_profile() \
                   .get_intrinsics()
    K = np.array([[intr.fx, 0.0,     intr.ppx],
                  [0.0,     intr.fy, intr.ppy],
                  [0.0,     0.0,     1.0]])
    return K, np.array(intr.coeffs, dtype=float)


# ---------------------------------------------------------------------------
# Utilitaires depth
# ---------------------------------------------------------------------------

def depth_to_colormap(depth_image: np.ndarray,
                      depth_scale: float = 0.001,
                      max_range_m: float = 0.6) -> np.ndarray:
    """
    Convertit une image de profondeur (uint16) en image couleur pour l'affichage
    (colormap TURBO : proche = bleu, loin = rouge ; 0 m → 'max_range_m' mètres).
    Les pixels invalides (0) sont affichés en noir.

    Returns:
        ndarray uint8 (H, W, 3) BGR
    """
    import cv2
    depth_m  = depth_image.astype(np.float32) * depth_scale
    depth_8  = np.clip(depth_m / max_range_m * 255.0, 0, 255).astype(np.uint8)
    out      = cv2.applyColorMap(depth_8, cv2.COLORMAP_TURBO)
    out[depth_image == 0] = 0
    return out
