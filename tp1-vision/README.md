# TP1 Vision — dépôt enseignant

```
eleves/    ce que reçoivent les élèves (source unique, copiée telle quelle dans le zip)
prof/      jamais distribué : solutions, génération de l'exemple, version corrigée, tests
build.py   assemble dist/ et fabrique le zip
dist/      généré (ignoré par git)
```

## Structure détaillée

```
eleves/
├── README.md  requirements.txt
├── ma_vision.py          gabarit : 9 énoncés, JAMAIS rempli ici (le build vérifie qu'il n'y a aucune solution)
├── camera_utils.py       partagé (notebook et nœud)
├── captures/             exemple.png, exemple_depth.npy, exemple.json  (à copier ici : non fournis dans cette archive)
├── notebook/             tp1_vision.py  exercices.py  checks.py   (+ mes_reponses.py créé au lancement)
└── live/                 vision_node.py  zmq_utils.py  zmq_test_pub.py  zmq_test_sub.py
prof/
├── vision_ref.py         solutions de référence (+ process_frame)
├── ma_vision_corrige.py  module élève « terminé » (sert aux tests du nœud)
├── make_sample.py        régénère captures/exemple.* (lit prof/_scene_src.png, non fourni)
└── tests/                e2e_notebook.py  e2e_node.py
```

## Travailler sur le TP (développement)

Le notebook et le nœud importent `vision_ref`, qui n'est volontairement pas dans `eleves/` :
on le rend visible avec `PYTHONPATH`.

```bash
export PYTHONPATH=$PWD/prof
marimo edit eleves/notebook/tp1_vision.py
python eleves/notebook/checks.py vision_ref        # doit afficher « Tout est vert »
python eleves/live/vision_node.py --replay         # nœud sans caméra (exige process_frame exportée dans ma_vision.py)
```

Attention : le bouton « Exporter » du notebook écrit dans `eleves/ma_vision.py`. Pour tester comme un élève,
travaillez dans une copie (dézippez `dist/`), pas dans `eleves/`. Si vous avez exporté par erreur :
`git checkout eleves/ma_vision.py`.

## Construire le zip

```bash
python build.py --version 2026-27                  # versions de Python 3.10 à 3.14 trouvées sur la machine
python build.py --pythons 3.11 3.12 --version 2026-27
```

Le script : vérifie la référence, vérifie que `ma_vision.py` ne contient aucune solution, copie `eleves/` dans
`dist/tp1_vision/`, compile `prof/vision_ref.py` en `_vref_<version>.pyc` (une fois par version de Python
installée : `python3.11`, `python3.12`…), écrit le chargeur `vision_ref.py`, revérifie, puis zippe.
Les `.pyc` ne dépendent pas du système d'exploitation, seulement de la version de Python.

Équivalent à la main, pour une version donnée :

```bash
python3.12 -c "import py_compile; py_compile.compile('prof/vision_ref.py', cfile='dist/tp1_vision/_vref_312.pyc', optimize=2)"
```

## Tests de bout en bout (`pip install playwright && playwright install chromium`)

```bash
python prof/tests/e2e_notebook.py --captures /chemin/captures   # 27 éditeurs, verrous, export, checks.py
python prof/tests/e2e_node.py     --captures /chemin/captures   # nœud --replay + Subscriber ZeroMQ
python prof/tests/e2e_notebook.py --target dist/tp1_vision      # même test sur le dossier distribué
```
(`--captures` inutile si `eleves/captures/exemple.*` est présent.)

## Versions

Un tag par année distribuée : `git tag v2026-27` juste après avoir généré le zip.
