"""
vision_node.py — Nœud de perception vision temps réel
=======================================================
Ce nœud autonome
  1. capture les frames couleur + profondeur de la D405 (ou rejoue une capture),
  2. appelle VOTRE `process_frame` (ma_vision.py, écrite et exportée depuis le notebook) :
     détection de tous les objets de la couleur choisie, avec centre, profondeur et position 3D,
  3. publie le résultat sur ZeroMQ (topic "vision/perception").

Rien à compléter ici pour que ça marche. La boucle de `run()` est volontairement courte :
repérez la zone « 🔧 à vous » si vous voulez filtrer ou transformer les objets avant de les publier.

Usage (depuis le dossier du TP) :
    python live/vision_node.py                 # caméra D405 + fenêtre : 4 vues et curseurs (q pour quitter)
    python live/vision_node.py --replay        # sans caméra : rejoue captures/exemple.*
    python live/vision_node.py --no-show       # sans fenêtre (robot, SSH, serveur)

Le nœud exige que `process_frame` (et les fonctions qu'elle appelle) soit écrite et exportée dans
ma_vision.py : sinon il s'arrête avec un message qui dit quoi faire.
"""

import argparse
import json
import sys
import time
from pathlib import Path

# le dossier du TP (parent de live/) contient ma_vision.py, camera_utils.py, captures/, vision_ref
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os

import cv2
import numpy as np

from zmq_utils import Publisher


def _fix_qt_fonts():
    """
    Linux : la fenêtre OpenCV (Qt) n'affiche ni le nom ni la valeur des curseurs si elle ne trouve pas de police.
    Les paquets pip d'OpenCV forcent QT_QPA_FONTDIR vers un dossier vide au moment de `import cv2` : on le
    redirige donc APRÈS l'import, vers les polices du système (à installer si besoin : sudo apt install fonts-dejavu-core).
    """
    if not sys.platform.startswith("linux"):
        return
    own = os.path.join(os.path.dirname(cv2.__file__), "qt", "fonts")
    if os.path.isdir(own) and os.listdir(own):
        return                                                   # OpenCV embarque ses polices : rien à faire
    for d in ("/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/dejavu", "/usr/share/fonts/TTF",
              "/usr/share/fonts/truetype", "/usr/share/fonts"):
        if os.path.isdir(d):
            os.environ["QT_QPA_FONTDIR"] = d
            return
    print("[NODE] ⚠ aucune police système trouvée : les curseurs n'auront pas de texte. "
          "Installez-en une : sudo apt install fonts-dejavu-core")


_fix_qt_fonts()

import ma_vision as V          # vos fonctions (exportées depuis le notebook)


# Paramètres HSV par défaut (objet orange) — à reporter depuis le notebook (section 3) ou depuis la fenêtre du nœud
DEFAULT_HSV = {
    "h_min": 5,  "h_max": 20,
    "s_min": 120, "s_max": 255,
    "v_min": 80,  "v_max": 255,
}
MIN_AREA = 500          # pixels² : en dessous, c'est du bruit
K_OPEN, K_CLOSE = 7, 15  # motifs d'ouverture / fermeture (pixels)


# ===========================================================================
# Affichage (fourni)
# ===========================================================================

def draw_overlay(frame: np.ndarray, objects: list[dict]) -> np.ndarray:
    """Dessine contour, centre et position de chaque objet. Fourni."""
    out = frame.copy()
    for i, o in enumerate(objects):
        u, v = (int(round(t)) for t in o["centroid"])
        X, Y, Z = o["xyz_m"]
        cv2.drawContours(out, [o["contour"]], -1, (0, 255, 0), 2)
        cv2.circle(out, (u, v), 5, (0, 255, 0), -1)
        cv2.putText(out, f"#{i}  Z={Z:.3f} m", (u + 10, v - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return out


TRACKBARS = [("H min", "h_min", 179), ("H max", "h_max", 179), ("S min", "s_min", 255), ("S max", "s_max", 255),
             ("V min", "v_min", 255), ("V max", "v_max", 255),
             ("Ouverture", "k_open", 31), ("Fermeture", "k_close", 51), ("Aire min", "min_area", 5000)]


def make_window(window_name: str, params: dict):
    """Fenêtre OpenCV avec les curseurs HSV, ouverture, fermeture et aire minimale. Fourni."""
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 800)
    for label, key, vmax in TRACKBARS:
        cv2.createTrackbar(label, window_name, params[key], vmax, lambda x: None)


def read_trackbars(window_name: str) -> dict:
    """Lit les curseurs. Les motifs sont forcés impairs (≥ 1). Fourni."""
    p = {k: cv2.getTrackbarPos(l, window_name) for l, k, _ in TRACKBARS}
    p["k_open"], p["k_close"] = max(1, p["k_open"] | 1), max(1, p["k_close"] | 1)
    return p


def build_grid(color, depth, depth_scale, objects, hsv_range, params):
    """
    Grille 2×2 : [RGB + détections] | [profondeur] / [RGB filtré HSV + bornes] | [profondeur masquée].
    Fourni.
    """
    from_cam = _depth_colormap(depth, depth_scale)
    H, W = color.shape[:2]
    lo = (hsv_range["h_min"], hsv_range["s_min"], hsv_range["v_min"])
    hi = (hsv_range["h_max"], hsv_range["s_max"], hsv_range["v_max"])
    mask = cv2.inRange(cv2.cvtColor(color, cv2.COLOR_BGR2HSV), np.array(lo, np.uint8), np.array(hi, np.uint8))   # masque brut

    top_left  = draw_overlay(color, objects)
    top_right = from_cam
    bot_left  = cv2.bitwise_and(color, color, mask=mask)
    bot_right = cv2.bitwise_and(from_cam, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))
    grid = np.vstack([np.hstack([top_left, top_right]), np.hstack([bot_left, bot_right])])

    for text, pos in [("RGB + detections", (10, 25)), ("Profondeur", (W + 10, 25)),
                      ("Filtre HSV", (10, H + 25)), ("Profondeur masquee", (W + 10, H + 25))]:
        cv2.putText(grid, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(grid, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
    return grid


def _depth_colormap(depth, depth_scale):
    """Colormap de profondeur (indépendante de la caméra, pour fonctionner en replay)."""
    d8 = np.clip(depth.astype(np.float32) * depth_scale / 0.6 * 255, 0, 255).astype(np.uint8)
    out = cv2.applyColorMap(d8, cv2.COLORMAP_TURBO)
    out[depth == 0] = 0
    return out


# ===========================================================================
# Sources d'images : caméra ou rejeu d'une capture
# ===========================================================================

class ReplaySource:
    """Rejoue indéfiniment captures/<nom>.png + _depth.npy + .json (pas de caméra requise)."""

    def __init__(self, base=None):
        b = Path(base) if base else ROOT / "captures" / "exemple"
        self.color = cv2.imread(str(b.with_suffix(".png")))
        self.depth = np.load(str(b) + "_depth.npy")
        info = json.loads(b.with_suffix(".json").read_text())
        self.K = np.array([[info["fx"], 0, info["ppx"]], [0, info["fy"], info["ppy"]], [0, 0, 1]], float)
        self.depth_scale = float(info["depth_scale"])
        if self.color is None:
            raise FileNotFoundError(f"{b}.png introuvable")
        print(f"[REPLAY] {b}  ({self.color.shape[1]}×{self.color.shape[0]})")

    def read(self):
        time.sleep(1 / 30)
        return self.color, self.depth

    def close(self):
        pass


class CameraSource:
    """Caméra D405 (import de camera_utils seulement ici : pyrealsense2 n'est pas requis en replay)."""

    def __init__(self):
        import camera_utils as cam
        self.cam = cam
        self.pipeline, self.align = cam.init_d405()
        self.K, _ = cam.get_intrinsics(self.pipeline)
        self.depth_scale = cam.get_depth_scale(self.pipeline)
        print(f"[D405] depth_scale = {self.depth_scale}")

    def read(self):
        return self.cam.get_aligned_frames(self.pipeline, self.align)

    def close(self):
        self.cam.stop_d405(self.pipeline)


# ===========================================================================
# Boucle principale & publication ZeroMQ
# ===========================================================================

def build_payload(objects, frame_count):
    """Message publié : une entrée par objet (tout en types simples, sérialisables)."""
    return {
        "timestamp":   time.time(),
        "frame_count": frame_count,
        "objects": [
            {"id": i,
             "centroid": [float(o["centroid"][0]), float(o["centroid"][1])],
             "area":     float(o["area"]),
             "depth_m":  float(o["depth_m"]),
             "xyz_m":    [float(x) for x in o["xyz_m"]]}
            for i, o in enumerate(objects)
        ],
    }


def get_objects(color, depth, src, hsv_range, params):
    """Appelle VOTRE process_frame (ma_vision.py). S'arrête avec un message clair si elle n'est pas prête."""
    try:
        objects = V.process_frame(color, depth, src.K, src.depth_scale, hsv_range,
                                  params["min_area"], params["k_open"], params["k_close"])
    except NotImplementedError:
        sys.exit("[NODE] ✋ process_frame (ou une des fonctions qu'elle appelle) n'est pas écrite dans ma_vision.py.\n"
                 "       Écrivez-la dans le notebook puis cliquez sur « Exporter » (section 8).\n"
                 "       Vérification : python notebook/checks.py")
    if objects is None:
        sys.exit("[NODE] ✋ process_frame a renvoyé None : pensez au `return` (liste d'objets, vide s'il n'y en a pas).")
    return objects


def run(args):
    pub = Publisher(port=args.port, topic="vision/perception")
    src = ReplaySource() if args.replay else CameraSource()

    WIN = "Vision Node"
    params = {**DEFAULT_HSV, "k_open": K_OPEN, "k_close": K_CLOSE, "min_area": MIN_AREA}
    if args.show:
        make_window(WIN, params)

    print("Nœud de perception démarré. Ctrl+C pour arrêter" + (" (ou q dans la fenêtre)." if args.show else "."))
    frame_count, t_start = 0, time.time()
    try:
        while True:
            color, depth = src.read()
            if args.show:
                params = read_trackbars(WIN)                         # curseurs de la fenêtre
            hsv_range = {k: params[k] for k in DEFAULT_HSV}

            objects = get_objects(color, depth, src, hsv_range, params)

            # 🔧 à vous : ici `objects` est la liste de tous les objets de l'image.
            #    Vous pouvez la filtrer (distance maximale, zone utile…) ou la trier avant de la publier.

            pub.send(build_payload(objects, frame_count))

            if args.show:
                cv2.imshow(WIN, build_grid(color, depth, src.depth_scale, objects, hsv_range, params))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_count += 1
            if frame_count % 30 == 0:
                fps = frame_count / (time.time() - t_start)
                print(f"[NODE] {frame_count} frames  |  {fps:.1f} fps  |  {len(objects)} objet(s)")
    except KeyboardInterrupt:
        print("\n[NODE] Arrêt.")
    finally:
        src.close()
        pub.close()
        if args.show:
            cv2.destroyAllWindows()
            if params != {**DEFAULT_HSV, "k_open": K_OPEN, "k_close": K_CLOSE, "min_area": MIN_AREA}:
                print("\nRéglages de la fenêtre, à recopier en haut de vision_node.py :")
                print("DEFAULT_HSV = " + json.dumps({k: params[k] for k in DEFAULT_HSV}))
                print(f"MIN_AREA = {params['min_area']}")
                print(f"K_OPEN, K_CLOSE = {params['k_open']}, {params['k_close']}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Nœud de perception vision TP1")
    p.add_argument("--port", type=int, default=5555, help="Port ZeroMQ de publication (défaut: 5555)")
    p.add_argument("--no-show", dest="show", action="store_false", help="Sans fenêtre (robot, SSH, serveur)")
    p.add_argument("--replay", action="store_true", help="Sans caméra : rejoue captures/exemple.*")
    run(p.parse_args())
