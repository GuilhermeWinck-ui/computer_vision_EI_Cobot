#!/usr/bin/env python3
"""
build.py — assemble le dossier / le zip distribué aux élèves
=============================================================
    python build.py                        # versions de Python : 3.10 à 3.14 (celles installées)
    python build.py --pythons 3.11 3.12    # seulement celles-ci
    python build.py --version 2026-27      # suffixe du zip : tp1_vision_eleves_2026-27.zip

Ce que fait le script (bibliothèque standard uniquement) :
  1. vérifie que la référence (prof/vision_ref.py) passe tous les tests ;
  2. vérifie que eleves/ma_vision.py ne contient que des énoncés (aucune solution) ;
  3. copie eleves/ dans dist/tp1_vision/ (sans mes_reponses.json, __pycache__, etc.) ; avertit si
     captures/ contient autre chose que exemple.* ;
  4. compile prof/vision_ref.py en dist/tp1_vision/_vref_<version>.pyc, une fois par version
     de Python trouvée (python3.11, python3.12…) ;
  5. écrit le chargeur dist/tp1_vision/vision_ref.py, qui choisit le bon .pyc ;
  6. vérifie le résultat (python notebook/checks.py vision_ref depuis dist/) ;
  7. fabrique dist/tp1_vision_eleves[_version].zip.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ELEVES, PROF, DIST = ROOT / "eleves", ROOT / "prof", ROOT / "dist"
OUT_NAME = "tp1_vision"
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", "mes_reponses.json", "__marimo__", ".DS_Store")

LOADER = '''"""Chargeur des solutions de référence (compilées, une par version de Python)."""
import importlib
import importlib.util
import sys

_name = f"_vref_{sys.version_info[0]}{sys.version_info[1]}"
if importlib.util.find_spec(_name) is None:
    raise ImportError(f"Pas de solution compilée pour Python {sys.version_info[0]}.{sys.version_info[1]} : "
                      "demandez le fichier à l'enseignant.")
sys.modules[__name__] = importlib.import_module(_name)
'''


def step(msg):
    print(f"\n▶ {msg}")


def fail(msg):
    print(f"❌ {msg}")
    sys.exit(1)


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def check_reference():
    step("Référence : python notebook/checks.py vision_ref")
    env = {**os.environ, "PYTHONPATH": str(PROF)}
    r = run([sys.executable, str(ELEVES / "notebook" / "checks.py"), "vision_ref"], cwd=ELEVES, env=env)
    if r.returncode != 0 or "Tout est vert" not in r.stdout:
        print(r.stdout[-1500:], r.stderr[-1500:])
        fail("la référence ne passe pas tous les tests")
    print("✅ référence entièrement verte")


def check_template_has_no_solution():
    step("eleves/ma_vision.py : énoncés seulement")
    text = (ELEVES / "ma_vision.py").read_text(encoding="utf-8")
    blocks = re.findall(r"# >>> exercice: (\w+)\n(.*?)# <<< exercice: \1\n", text, re.S)
    if not blocks:
        fail("aucun bloc `# >>> exercice` dans eleves/ma_vision.py")
    for name, body in blocks:
        if "raise NotImplementedError" not in body:
            fail(f"ma_vision.py : `{name}` contient autre chose que l'énoncé (solution oubliée ?)")
    print(f"✅ {len(blocks)} énoncés, aucune solution")


def copy_students_tree(target):
    step(f"Copie de eleves/ → {target.relative_to(ROOT)}")
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(ELEVES, target, ignore=IGNORE)
    caps = target / "captures"
    if not (caps / "exemple.png").exists():
        print("⚠️  eleves/captures/exemple.png est absent : l'image d'exemple manquera dans le zip.")
    others = sorted(p.name for p in caps.iterdir() if p.is_file() and not p.name.startswith("exemple") and p.name != ".gitkeep")
    if others:
        print(f"⚠️  eleves/captures/ contient d'autres fichiers que exemple.* : ils seront distribués aux élèves :\n    "
              + "\n    ".join(others)
              + "\n    (rangez-les plutôt dans prof/captures_test/)")
        if not STRICT_OK:
            if input("    Continuer quand même ? [o/N] ").strip().lower() not in ("o", "oui", "y"):
                fail("abandon : déplacez ces fichiers hors de eleves/captures/")


def compile_reference(target, versions):
    step("Compilation de prof/vision_ref.py")
    done, missing = [], []
    code = ("import py_compile, sys; py_compile.compile(sys.argv[1], cfile=sys.argv[2], doraise=True, optimize=2)")
    for v in versions:
        exe = shutil.which(f"python{v}")
        if exe is None and v == f"{sys.version_info[0]}.{sys.version_info[1]}":
            exe = sys.executable
        if exe is None:
            missing.append(v)
            continue
        cfile = target / f"_vref_{v.replace('.', '')}.pyc"
        r = run([exe, "-c", code, str(PROF / "vision_ref.py"), str(cfile)])
        if r.returncode != 0:
            fail(f"compilation Python {v} : {r.stderr[-500:]}")
        done.append(v)
        print(f"✅ Python {v} → {cfile.name}")
    if not done:
        fail("aucune version de Python trouvée")
    if missing:
        print(f"⚠️  versions non trouvées (pas de .pyc pour elles) : {', '.join(missing)}")
    (target / "vision_ref.py").write_text(LOADER, encoding="utf-8")
    return done


def verify(target, versions):
    step("Vérification du dossier généré")
    perso = list(target.rglob("mes_reponses.json"))
    if perso:
        fail(f"fichier personnel dans le dossier généré : {perso}")
    leaked = [p for p in target.rglob("*.py") if "def _ref_pixel" in p.read_text(encoding="utf-8", errors="ignore")]
    if leaked:
        fail(f"solutions en clair dans : {leaked}")
    cur = f"{sys.version_info[0]}.{sys.version_info[1]}"
    v = cur if cur in versions else versions[0]                   # un interpréteur pour lequel un .pyc existe
    exe = sys.executable if v == cur else shutil.which(f"python{v}")
    r = run([exe, str(target / "notebook" / "checks.py"), "vision_ref"], cwd=target)
    if r.returncode != 0 or "Tout est vert" not in r.stdout:
        print(r.stdout[-1500:], r.stderr[-1500:])
        fail("le dossier généré ne passe pas checks.py vision_ref (chargeur / .pyc)")
    print("✅ aucune solution en clair, pas de mes_reponses.json ; checks.py vision_ref vert avec le .pyc de cette version")
    for cache in target.rglob("__pycache__"):          # créés par la vérification : à ne pas distribuer
        shutil.rmtree(cache, ignore_errors=True)


def make_zip(target, version):
    name = f"{OUT_NAME}_eleves" + (f"_{version}" if version else "") + ".zip"
    zpath = DIST / name
    if zpath.exists():
        zpath.unlink()
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(target.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:
                z.write(p, Path(OUT_NAME) / p.relative_to(target))
    print(f"\n📦 {zpath.relative_to(ROOT)}  ({zpath.stat().st_size / 1024:.0f} Ko)")


STRICT_OK = False        # True avec --oui : ne pas poser la question (utile en automatique)


def main():
    global STRICT_OK
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pythons", nargs="+", default=["3.10", "3.11", "3.12", "3.13", "3.14"],
                    help="versions de Python des élèves (défaut : 3.10 à 3.14)")
    ap.add_argument("--version", default="", help="suffixe du zip, ex. 2026-27")
    ap.add_argument("--oui", action="store_true", help="continuer sans poser de question si captures/ contient autre chose que exemple.*")
    args = ap.parse_args()
    STRICT_OK = args.oui

    check_reference()
    check_template_has_no_solution()
    DIST.mkdir(exist_ok=True)
    target = DIST / OUT_NAME
    copy_students_tree(target)
    done = compile_reference(target, args.pythons)
    verify(target, done)
    make_zip(target, args.version)


if __name__ == "__main__":
    main()
