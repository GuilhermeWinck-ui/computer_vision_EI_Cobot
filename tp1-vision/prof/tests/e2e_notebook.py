#!/usr/bin/env python3
"""
e2e_notebook.py — test de bout en bout du notebook, dans un navigateur invisible (Playwright)
=============================================================================================
Fait ce que fait un élève : tape la solution dans chacun des 28 éditeurs, vérifie que les
résultats cachés (🔒) se débloquent, clique sur « Exporter », puis lance
`python notebook/checks.py ma_vision` sur le fichier exporté (doit être entièrement vert).

    python prof/tests/e2e_notebook.py                       # teste eleves/
    python prof/tests/e2e_notebook.py --target dist/tp1_vision   # teste le dossier généré
    python prof/tests/e2e_notebook.py --captures /chemin/vers/captures   # si eleves/captures/exemple.* manque

Le test travaille sur une COPIE temporaire : eleves/ma_vision.py et mes_reponses.json ne sont pas touchés.
Prérequis : pip install playwright && playwright install chromium
"""
import argparse
import importlib.util
import inspect
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PROF = REPO / "prof"


def load_reference():
    """La référence en SOURCE (prof/vision_ref.py), même si le dossier testé n'a que des .pyc."""
    spec = importlib.util.spec_from_file_location("vision_ref_src", PROF / "vision_ref.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def solutions(ref):
    fs = lambda n: inspect.getsource(getattr(ref, n))
    return {
        0: "b, g, r = img[pv, pu]\n", 1: "rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\n",
        2: "hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)\n", 3: "H, S, V = cv2.split(hsv)\n",
        4: "mask = cv2.inRange(hsv, lo, hi)\n", 5: "objet = cv2.bitwise_and(img, img, mask=mask)\n",
        6: "opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open)))\n",
        7: "closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close)))\n",
        8: fs("hsv_mask"), 9: fs("clean_mask"),
        10: "contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)\n",
        11: "objects = sorted([c for c in contours if cv2.contourArea(c) >= min_area], key=cv2.contourArea, reverse=True)\n",
        12: fs("find_objects"),
        13: "M = cv2.moments(c_obj)\n",
        14: "u_c, v_c = None, None\nif M['m00'] > 0:\n    u_c = M['m10'] / M['m00']\n    v_c = M['m01'] / M['m00']\n",
        15: fs("centroid"),
        16: "z_px = depth[pv, pu] * depth_scale\n",
        17: "obj_mask = np.zeros_like(mask_clean)\ncv2.drawContours(obj_mask, [c_obj], -1, 255, cv2.FILLED)\n",
        18: "valid = (obj_mask > 0) & (depth > 0)\nz_obj = float(np.median(depth[valid])) * depth_scale\n",
        19: "X = z_obj * (u_c - K[0, 2]) / K[0, 0]\nY = z_obj * (v_c - K[1, 2]) / K[1, 1]\nZ = z_obj\n",
        20: fs("object_depth"), 21: fs("deproject"),
        22: "gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)\ndetector = cv2.aruco.ArucoDetector(aruco_dict, params)\ncorners, ids, rejected = detector.detectMarkers(gray)\n",
        23: "L = marker_len\nobj = np.array([[-L/2, L/2, 0], [L/2, L/2, 0], [L/2, -L/2, 0], [-L/2, -L/2, 0]], np.float32)\n",
        24: "ok, rvec, tvec = cv2.solvePnP(obj, img_pts, K, dist_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE)\n",
        25: fs("detect_aruco"), 26: fs("marker_pose"), 27: fs("process_frame"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=str(REPO / "eleves"), help="dossier à tester (eleves/ ou dist/tp1_vision)")
    ap.add_argument("--captures", default=None, help="dossier contenant exemple.png / _depth.npy / .json")
    ap.add_argument("--port", type=int, default=2850)
    args = ap.parse_args()

    from playwright.sync_api import sync_playwright

    ref = load_reference()
    sol = solutions(ref)
    tmp = Path(tempfile.mkdtemp(prefix="e2e_nb_"))
    work = tmp / "tp1"
    shutil.copytree(args.target, work, ignore=shutil.ignore_patterns("__pycache__", "mes_reponses.json"))
    if args.captures:
        shutil.copytree(args.captures, work / "captures", dirs_exist_ok=True)
    if not (work / "captures" / "exemple.png").exists():
        sys.exit("❌ captures/exemple.png introuvable : utilisez --captures")
    env = {**os.environ}
    if not (work / "vision_ref.py").exists():                      # mode développement : source de la référence
        env["PYTHONPATH"] = str(PROF) + os.pathsep + env.get("PYTHONPATH", "")
    srv = subprocess.Popen([sys.executable, "-m", "marimo", "run", "notebook/tp1_vision.py", "--headless",
                            "--no-token", "--port", str(args.port)], cwd=work, env=env,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    ok = True
    try:
        time.sleep(8)
        with sync_playwright() as p:
            b = p.chromium.launch()
            pg = b.new_page(viewport={"width": 1400, "height": 1000})
            pg.goto(f"http://localhost:{args.port}")
            pg.wait_for_timeout(15000)
            count = lambda t: pg.get_by_text(t, exact=False).count()
            locks = lambda: count("afficheront") + count("s'affichera")
            n_ed = pg.locator(".cm-content").count()
            print(f"éditeurs : {n_ed} (attendu 28) · verrous au départ : {locks()}")
            ok &= n_ed == 28
            for i in range(28):
                ed = pg.locator(".cm-content").nth(i)
                ed.scroll_into_view_if_needed()
                ed.click()
                pg.keyboard.press("Control+A")
                pg.keyboard.insert_text(sol[i])
                pg.mouse.click(1395, 8)                              # clic en dehors : lance les tests
                pg.wait_for_timeout(1800)
                err, part = count("ne s'exécute pas"), count("tests réussis")
                print(f"  éditeur {i:2d} · verrous {locks():2d} · erreurs {err} · partiels {part}")
                ok &= (err == 0 and part == 0)
            ok &= locks() == 0
            btn = pg.get_by_role("button", name="💾 Exporter dans ma_vision.py")
            btn.scroll_into_view_if_needed()
            btn.click()
            pg.wait_for_timeout(3000)
            exported = count("mis à jour") > 0
            print("export :", "✅" if exported else "❌")
            ok &= exported
            b.close()
    finally:
        srv.terminate()
    r = subprocess.run([sys.executable, "notebook/checks.py", "ma_vision"], cwd=work, env=env,
                       capture_output=True, text=True)
    green = "Tout est vert" in r.stdout
    print("checks.py ma_vision (fichier exporté) :", "✅ tout vert" if green else "❌\n" + r.stdout[-1200:])
    ok &= green
    shutil.rmtree(tmp, ignore_errors=True)
    print("\nRÉSULTAT :", "✅ OK" if ok else "❌ ÉCHEC")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
