#!/usr/bin/env python3
"""
e2e_node.py — test de live/vision_node.py en --replay, avec un Subscriber ZeroMQ
==================================================================================
Lance le nœud sans caméra (rejeu de captures/exemple.*), reçoit un message et vérifie
la liste `objects` (centre, aire, profondeur, position 3D).

    python prof/tests/e2e_node.py                            # teste eleves/ avec prof/ma_vision_corrige.py
    python prof/tests/e2e_node.py --target dist/tp1_vision
    python prof/tests/e2e_node.py --captures /chemin/vers/captures

Travaille sur une copie temporaire : rien n'est modifié dans le dépôt.
"""
import argparse
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PROF = REPO / "prof"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=str(REPO / "eleves"))
    ap.add_argument("--captures", default=None)
    ap.add_argument("--port", type=int, default=5651)
    args = ap.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="e2e_node_"))
    work = tmp / "tp1"
    shutil.copytree(args.target, work, ignore=shutil.ignore_patterns("__pycache__", "mes_reponses.json"))
    if args.captures:
        shutil.copytree(args.captures, work / "captures", dirs_exist_ok=True)
    env = {**os.environ}
    if not (work / "vision_ref.py").exists():
        env["PYTHONPATH"] = str(PROF) + os.pathsep + env.get("PYTHONPATH", "")
    sys.path[:0] = [str(work), str(work / "live")]
    from zmq_utils import Subscriber                                          # noqa: E402

    ok = True
    scenarios = [("process_frame écrite (module corrigé)", PROF / "ma_vision_corrige.py", False),
                 ("process_frame non écrite (stubs) → le nœud s'arrête avec un message", None, True)]
    for label, module, expect_stop in scenarios:
        if module is not None:
            shutil.copy(module, work / "ma_vision.py")
        else:
            shutil.copy(REPO / "eleves" / "ma_vision.py", work / "ma_vision.py")     # énoncés de départ
        node = subprocess.Popen([sys.executable, "live/vision_node.py", "--replay", "--no-show", "--port", str(args.port)],
                                cwd=work, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        time.sleep(1.5)
        if expect_stop:                                                          # le nœud doit s'arrêter seul
            try:
                out = node.communicate(timeout=20)[0]
            except subprocess.TimeoutExpired:
                node.kill()
                out = node.communicate()[0]
            good = node.returncode not in (0, None) and "n'est pas écrite" in out
            print(f"{'✅' if good else '❌'} {label}")
            if not good:
                print(out[-800:])
            ok &= good
            args.port += 1
            continue
        sub = Subscriber(port=args.port, topic="vision/perception")
        msg = None
        for _ in range(30):
            msg = sub.receive(500)
            if msg:
                break
        sub.close()
        node.send_signal(signal.SIGINT)                                          # Ctrl+C : arrêt propre
        try:
            out = node.communicate(timeout=30)[0]
        except subprocess.TimeoutExpired:
            node.kill()
            out = node.communicate()[0]
        objs = (msg or {}).get("objects", [])
        good = bool(objs) and all(set(o) == {"id", "centroid", "area", "depth_m", "xyz_m"} for o in objs) \
            and all(len(o["xyz_m"]) == 3 and o["depth_m"] > 0 for o in objs) \
            and [o["area"] for o in objs] == sorted((o["area"] for o in objs), reverse=True)
        print(f"{'✅' if good else '❌'} {label} : {len(objs)} objet(s)")
        if not good:
            print(out[-800:])
        ok &= good
        args.port += 1
    shutil.rmtree(tmp, ignore_errors=True)
    print("\nRÉSULTAT :", "✅ OK" if ok else "❌ ÉCHEC")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
