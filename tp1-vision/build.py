#!/usr/bin/env python3
"""
build.py — assemble le dossier / le zip distribué aux élèves
=============================================================
    python build.py                        # versions de Python : 3.10 à 3.14 (celles installées)
    python build.py --pythons 3.11 3.12    # seulement celles-ci
    python build.py --version 2026-27      # suffixe du zip : tp1_vision_eleves_2026-27.zip
    python build.py --online               # VERSION EN LIGNE (molab) : dossier dist/tp1_online/ de 4 fichiers + captures/

Ce que fait le script (bibliothèque standard uniquement) :
  1. vérifie que la référence (prof/vision_ref.py) passe tous les tests ;
  2. vérifie que eleves/ma_vision.py ne contient que des énoncés (aucune solution) ;
  3. copie eleves/ dans dist/tp1_vision/ (sans mes_reponses.*, __pycache__, etc.) ; avertit si
     captures/ contient autre chose que exemple.* ;
  4. compile prof/vision_ref.py en dist/tp1_vision/_vref_<version>.pyc, une fois par version
     de Python trouvée (python3.11, python3.12…) ;
  5. écrit le chargeur dist/tp1_vision/vision_ref.py, qui choisit le bon .pyc ;
  6. vérifie le résultat (python notebook/checks.py vision_ref depuis dist/) ;
  7. fabrique dist/tp1_vision_eleves[_version].zip.

Avec --online (au lieu des étapes 3 à 7) : dist/tp1_online/ contient notebook.py (le même notebook, avec l'en-tête
de dépendances et le titre de page en ligne), exercices.py, checks.py, vision_ref.py (bytecode intégré en texte, une
version par Python trouvé, car molab ne conserve que les .py) et captures/exemple.*. Le notebook s'adapte tout seul :
sans ma_vision.py à côté, il exporte par téléchargement.
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
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", "mes_reponses.*", "__marimo__", ".DS_Store")

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


ONLINE_DIR = DIST / "tp1_online"
ONLINE_HEADER = '''# /// script
# dependencies = [
#     "marimo",
#     "numpy",
#     "matplotlib",
#     "opencv-python-headless",
#     "anywidget",
# ]
# ///

'''
APP_LOCAL = 'app = marimo.App(width="full", app_title="TP1 — Vision par ordinateur")'
APP_ONLINE = 'app = marimo.App(width="medium", auto_download=["html"])'

ONLINE_REF_HEAD = '''"""vision_ref.py : solutions de référence (enseignant), distribuées compilées.

Le bytecode est intégré dans ce fichier texte (molab conserve les .py, pas les .pyc).
Versions de Python prises en charge : {versions}.
"""
import base64
import marshal
import sys
import zlib

_BLOBS = {{
'''
ONLINE_REF_TAIL = '''}

_tag = f"{sys.version_info[0]}{sys.version_info[1]}"
if _tag not in _BLOBS:
    raise ImportError(f"Pas de solution compilée pour Python {sys.version_info[0]}.{sys.version_info[1]} : "
                      "demandez le fichier à l'enseignant.")

exec(marshal.loads(zlib.decompress(base64.b64decode(_BLOBS[_tag]))), globals())
del _BLOBS, _tag
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
    perso = list(target.rglob("mes_reponses.*"))
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
    print("✅ aucune solution en clair, pas de mes_reponses ; checks.py vision_ref vert avec le .pyc de cette version")
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


def build_online(pythons):
    """Dossier dist/tp1_online/ : les 4 fichiers à mettre en ligne + captures/exemple.*"""
    step("Version en ligne : dist/tp1_online/")
    if ONLINE_DIR.exists():
        shutil.rmtree(ONLINE_DIR)
    ONLINE_DIR.mkdir(parents=True)

    # 1. notebook.py : le même notebook + en-tête de dépendances + titre de page en ligne
    src = (ELEVES / "notebook" / "tp1_vision.py").read_text(encoding="utf-8")
    if src.count(APP_LOCAL) != 1:
        fail("ligne `marimo.App(...)` du notebook introuvable ou modifiée : adaptez APP_LOCAL dans build.py")
    (ONLINE_DIR / "notebook.py").write_text(ONLINE_HEADER + src.replace(APP_LOCAL, APP_ONLINE), encoding="utf-8")
    shutil.copy(ELEVES / "notebook" / "exercices.py", ONLINE_DIR / "exercices.py")
    shutil.copy(ELEVES / "notebook" / "checks.py", ONLINE_DIR / "checks.py")
    print("✅ notebook.py, exercices.py, checks.py")

    # 2. vision_ref.py : bytecode (un par version de Python) intégré en texte
    code = ("import base64, marshal, sys, zlib; "
            "src = open(sys.argv[1], encoding='utf-8').read(); "
            "print(base64.b64encode(zlib.compress(marshal.dumps(compile(src, 'vision_ref.py', 'exec', optimize=2)), 9)).decode())")
    blobs, missing = {}, []
    for v in pythons:
        exe = shutil.which(f"python{v}")
        if exe is None and v == f"{sys.version_info[0]}.{sys.version_info[1]}":
            exe = sys.executable
        if exe is None:
            missing.append(v)
            continue
        r = run([exe, "-c", code, str(PROF / "vision_ref.py")])
        if r.returncode != 0:
            fail(f"compilation Python {v} : {r.stderr[-500:]}")
        blobs[v.replace(".", "")] = r.stdout.strip()
        print(f"✅ Python {v} compilé")
    if not blobs:
        fail("aucune version de Python trouvée")
    if missing:
        print(f"⚠️  versions non trouvées (pas de bytecode pour elles) : {', '.join(missing)}")
    body = "".join(f'    "{tag}": """\n' + "\n".join(b[i:i + 108] for i in range(0, len(b), 108)) + '\n""",\n' for tag, b in blobs.items())
    (ONLINE_DIR / "vision_ref.py").write_text(
        ONLINE_REF_HEAD.format(versions=", ".join(sorted(f"{t[0]}.{t[1:]}" for t in blobs))) + body + ONLINE_REF_TAIL, encoding="utf-8")

    # 3. captures/exemple.*
    caps = ONLINE_DIR / "captures"
    caps.mkdir()
    ex = sorted(ELEVES.glob("captures/exemple*"))
    for p in ex:
        shutil.copy(p, caps / p.name)
    if not ex:
        print("⚠️  eleves/captures/exemple.* est absent : à ajouter à la main dans captures/ en ligne.")

    # 4. vérifications
    step("Vérification du dossier en ligne")
    leaked = [p for p in ONLINE_DIR.rglob("*.py") if "def _ref_pixel" in p.read_text(encoding="utf-8", errors="ignore")]
    if leaked:
        fail(f"solutions en clair dans : {leaked}")
    cur = f"{sys.version_info[0]}.{sys.version_info[1]}"
    tag = cur.replace(".", "")
    v = cur if tag in blobs else sorted(blobs)[0]
    exe = sys.executable if v == cur or tag in blobs else shutil.which(f"python{v[0]}.{v[1:]}")
    r = run([exe, str(ONLINE_DIR / "checks.py"), "vision_ref"], cwd=ONLINE_DIR)
    if r.returncode != 0 or "Tout est vert" not in r.stdout:
        print(r.stdout[-1500:], r.stderr[-1500:])
        fail("le dossier en ligne ne passe pas checks.py vision_ref")
    for cache in ONLINE_DIR.rglob("__pycache__"):
        shutil.rmtree(cache, ignore_errors=True)
    print("✅ aucune solution en clair ; checks.py vision_ref vert avec le bytecode intégré")
    print(f"\n📁 {ONLINE_DIR.relative_to(ROOT)}/  → à remplacer dans molab : notebook.py, exercices.py, checks.py, vision_ref.py (+ captures/ si absent)")


STRICT_OK = False        # True avec --oui : ne pas poser la question (utile en automatique)


def main():
    global STRICT_OK
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pythons", nargs="+", default=["3.10", "3.11", "3.12", "3.13", "3.14"],
                    help="versions de Python des élèves (défaut : 3.10 à 3.14)")
    ap.add_argument("--version", default="", help="suffixe du zip, ex. 2026-27")
    ap.add_argument("--online", action="store_true", help="fabrique la version en ligne (molab) : dist/tp1_online/")
    ap.add_argument("--oui", action="store_true", help="continuer sans poser de question si captures/ contient autre chose que exemple.*")
    args = ap.parse_args()
    STRICT_OK = args.oui

    check_reference()
    if args.online:
        DIST.mkdir(exist_ok=True)
        build_online(args.pythons)
        return
    check_template_has_no_solution()
    DIST.mkdir(exist_ok=True)
    target = DIST / OUT_NAME
    copy_students_tree(target)
    done = compile_reference(target, args.pythons)
    verify(target, done)
    make_zip(target, args.version)


if __name__ == "__main__":
    main()
