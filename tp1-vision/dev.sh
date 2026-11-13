#!/usr/bin/env bash
# dev.sh — lance les outils du TP en mode développement (prof/ ajouté au chemin Python)
# Usage :   ./dev.sh notebook | nb | check | ref | node [options]
#   notebook   marimo edit .   (page d'accueil marimo : cliquez eleves > notebook > tp1_vision.py)
#   nb         ouvre directement eleves/notebook/tp1_vision.py
#   check      teste la référence (doit afficher « Tout est vert »)
#   node ...   eleves/live/vision_node.py avec vos options (ex. ./dev.sh node --replay --no-show)
#   ref        teste eleves/ma_vision.py (le module exporté)
# Si vous exportez depuis le notebook, ça modifie eleves/ma_vision.py : `git checkout eleves/ma_vision.py` ensuite.
set -e
cd "$(dirname "$0")"
export PYTHONPATH="$PWD/prof${PYTHONPATH:+:$PYTHONPATH}"
cmd="${1:-notebook}"; shift || true
case "$cmd" in
  notebook) exec marimo edit . "$@" ;;
  nb)       exec marimo edit eleves/notebook/tp1_vision.py "$@" ;;
  check)    cd eleves && exec python notebook/checks.py vision_ref ;;
  ref)      cd eleves && exec python notebook/checks.py ma_vision ;;
  node)     cd eleves && exec python live/vision_node.py "$@" ;;
  *) echo "Usage : ./dev.sh notebook | nb | check | ref | node [options]"; exit 1 ;;
esac
