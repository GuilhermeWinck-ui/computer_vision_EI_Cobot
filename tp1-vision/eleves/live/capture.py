#!/usr/bin/env python3
"""
capture.py — Flux de la caméra + sauvegarde de captures
========================================================
Affiche le flux couleur de la D405. Quand vous appuyez sur « s », l'image courante est
sauvegardée (couleur + profondeur + intrinsèques) dans le dossier `captures/` du TP, sous
un nom qui s'incrémente tout seul : capture_1, capture_2, capture_3…

    python live/capture.py                          # captures/capture_1, _2, ...
    python live/capture.py --path captures/piece    # captures/piece_1, _2, ...
    python live/capture.py --path /tmp/essai        # n'importe quel dossier

Touches :   s = sauvegarder     d = afficher / masquer la profondeur     q ou Échap = quitter

Chaque capture donne 3 fichiers, au format que le notebook sait relire :
    <nom>.png          image couleur
    <nom>_depth.npy    profondeur brute (uint16)
    <nom>.json         intrinsèques (fx, fy, ppx, ppy, dist) et depth_scale

Dans le notebook, choisissez ensuite le fichier <nom>.png comme image de travail.
Les numéros déjà pris ne sont jamais écrasés : en relançant le script, la numérotation continue.
"""

import argparse
import glob
import json
import os
import re
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                       # dossier du TP (celui qui contient camera_utils.py)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import camera_utils as cam                        # noqa: E402


def _fix_qt_fonts():
    """`import cv2` pointe Qt vers un dossier de polices vide : on le redirige vers celui du système."""
    for d in ("/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/dejavu", "/usr/share/fonts"):
        if os.path.isdir(d):
            os.environ["QT_QPA_FONTDIR"] = d
            return


_fix_qt_fonts()


def next_index(base):
    """Premier numéro libre pour `base` : base_1, base_2, ... (continue après les captures existantes)."""
    pat = re.compile(re.escape(os.path.basename(base)) + r"_(\d+)\.png$")
    taken = [int(m.group(1)) for f in glob.glob(base + "_*.png") if (m := pat.search(os.path.basename(f)))]
    return max(taken, default=0) + 1


def save_capture(base, idx, color, depth, K, dist, depth_scale):
    """Écrit base_<idx>.png, base_<idx>_depth.npy et base_<idx>.json ; renvoie le chemin du .png."""
    stem = f"{base}_{idx}"
    os.makedirs(os.path.dirname(stem) or ".", exist_ok=True)
    cv2.imwrite(stem + ".png", color)
    np.save(stem + "_depth.npy", depth.astype(np.uint16))
    meta = dict(width=int(color.shape[1]), height=int(color.shape[0]),
                fx=float(K[0, 0]), fy=float(K[1, 1]), ppx=float(K[0, 2]), ppy=float(K[1, 2]),
                dist=[float(x) for x in dist], depth_scale=float(depth_scale))
    with open(stem + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    return stem + ".png"


def main():
    ap = argparse.ArgumentParser(description="Flux caméra + sauvegarde d'images avec la touche s.")
    ap.add_argument("--path", default=os.path.join(ROOT, "captures", "capture"),
                    help="début du nom des fichiers (défaut : captures/capture → capture_1.png, capture_2.png…)")
    args = ap.parse_args()
    base = os.path.abspath(os.path.expanduser(args.path))

    pipeline, align = cam.init_d405()
    try:
        K, dist = cam.get_intrinsics(pipeline)
        depth_scale = cam.get_depth_scale(pipeline)
        idx = next_index(base)
        message, show_depth = "", False
        print(f"Touches : s = sauvegarder ({os.path.basename(base)}_{idx}…) | d = profondeur | q = quitter")

        while True:
            color, depth = cam.get_aligned_frames(pipeline, align)
            view = cam.depth_to_colormap(depth, depth_scale) if show_depth else color.copy()
            cv2.putText(view, message or f"s : sauvegarder  ->  {os.path.basename(base)}_{idx}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(view, message or f"s : sauvegarder  ->  {os.path.basename(base)}_{idx}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow("capture (s = sauvegarder, q = quitter)", view)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("d"):
                show_depth = not show_depth
            elif key == ord("s"):
                path = save_capture(base, idx, color, depth, K, dist, depth_scale)   # on sauvegarde l'image COULEUR, pas l'affichage
                print(f"✅ sauvegardé : {path}")
                message = f"sauvegarde : {os.path.basename(path)}"
                idx += 1
            elif key != 255:
                message = ""
    finally:
        cam.stop_d405(pipeline)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
