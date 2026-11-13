"""
checks.py — Tests automatiques des fonctions du TP1
====================================================
Chaque fonction `check_<nom>(fn)` exécute des tests sur des images SYNTHÉTIQUES
dont la bonne réponse est connue, et renvoie une liste de `Result`.

Utilisé par :
  - le notebook (tp1_vision.py) : bouton de vérification sous chaque exercice ;
  - le terminal, pour revérifier votre module avant de lancer le nœud :

        python notebook/checks.py            # teste ma_vision.py  (depuis le dossier du TP)
        python notebook/checks.py vision_ref # teste la référence (doit tout passer)
"""

import importlib
import os
import sys
from dataclasses import dataclass

import cv2
import numpy as np

# le dossier du TP (parent de notebook/) contient ma_vision.py, vision_ref, camera_utils
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import vision_ref


@dataclass
class Result:
    label: str          # ce qui est vérifié
    ok: object          # True / False / None (test sauté)
    detail: str = ""    # explication (surtout en cas d'échec)



def _contour_mask(contour, shape):
    """(tests) masque 0/255 de l'intérieur d'un contour."""
    m = np.zeros(shape[:2], np.uint8)
    cv2.drawContours(m, [contour], -1, 255, cv2.FILLED)
    return m


def _marker_object_points(L):
    """(tests) coins du marqueur dans son repère, plan z = 0."""
    return np.array([[-L / 2, L / 2, 0], [L / 2, L / 2, 0], [L / 2, -L / 2, 0], [-L / 2, -L / 2, 0]], np.float32)


def _expect(cond, msg):
    if not cond:
        raise AssertionError(msg)


def _run(tests):
    """tests = [(label, fonction_de_test, gate)]. Si un test « gate » échoue, les suivants sont sautés."""
    out, blocked = [], False
    for label, test, gate in tests:
        if blocked:
            out.append(Result(label, None, "sauté : corrigez d'abord le test précédent"))
            continue
        try:
            test()
            out.append(Result(label, True))
        except AssertionError as e:
            out.append(Result(label, False, str(e)))
            blocked = blocked or gate
        except NotImplementedError:
            out.append(Result(label, False, "fonction pas encore écrite (`raise NotImplementedError`)"))
            blocked = True
        except Exception as e:                               # erreur dans le code de l'étudiant
            out.append(Result(label, False, f"{type(e).__name__} : {e}"))
            blocked = blocked or gate
    return out


def summary(results):
    """(nb de tests réussis, nb total, tout est vert ?)"""
    n_ok = sum(1 for r in results if r.ok is True)
    return n_ok, len(results), all(r.ok is True for r in results)


# ---------------------------------------------------------------------------
# Outils de test
# ---------------------------------------------------------------------------

def _bgr(h, s, v):
    """Un pixel BGR dont la valeur HSV (convention OpenCV) est (h, s, v)."""
    return cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2BGR)[0, 0]


def _is_mask(m, shape):
    _expect(m is not None, "la fonction renvoie `None` : pensez au `return`")
    _expect(isinstance(m, np.ndarray), f"il faut renvoyer un tableau NumPy, pas un `{type(m).__name__}`")
    _expect(m.shape == shape, f"la forme doit être {shape} (une valeur par pixel), pas {m.shape}")
    _expect(m.dtype == np.uint8, f"le type doit être `uint8`, pas `{m.dtype}`")


# ---------------------------------------------------------------------------
# Exercice : hsv_mask(img_bgr, lo, hi)
# ---------------------------------------------------------------------------

def check_hsv_mask(fn):
    LO, HI = (5, 100, 100), (25, 255, 255)

    def two_colors():
        img = np.zeros((40, 60, 3), np.uint8)
        img[:, :30] = _bgr(15, 255, 255)        # orange
        img[:, 30:] = _bgr(120, 255, 255)       # bleu
        return img

    def t_form():
        _is_mask(fn(two_colors(), LO, HI), (40, 60))

    def t_values():
        m = fn(two_colors(), LO, HI)
        _expect(set(np.unique(m)) <= {0, 255}, f"le masque ne doit contenir que 0 et 255, il contient {sorted(set(np.unique(m)))}")

    def t_colors():
        m = fn(two_colors(), LO, HI)
        left, right = m[:, :30], m[:, 30:]
        if left.max() == 0 and right.min() == 255:
            raise AssertionError("c'est l'inverse : la zone **bleue** est gardée. Canaux B et R inversés ? "
                                 "OpenCV lit en **BGR** (`COLOR_BGR2HSV`).")
        _expect(left.max() > 0, "l'orange n'est pas détecté du tout : avez-vous converti avec `COLOR_BGR2HSV` ? (OpenCV lit en **BGR**, pas en RGB)")
        _expect((left == 255).all(), f"la zone orange (H = 15) devrait être entièrement à 255 : {int((left == 255).mean() * 100)} % seulement")
        _expect((right == 0).all(), "la zone bleue (H = 120) devrait être à 0")

    def t_bounds():
        img = np.zeros((1, 4, 3), np.uint8)
        for i, h in enumerate((4, 5, 25, 26)):
            img[0, i] = _bgr(h, 255, 255)
        got = [int(x) for x in fn(img, LO, HI)[0] // 255]
        _expect(got == [0, 1, 1, 0],
                f"pour H = 4, 5, 25, 26 on attend [0, 1, 1, 0] (bornes **incluses**), vous obtenez {got}")

    def t_saturation():
        img = np.zeros((1, 3, 3), np.uint8)
        img[0, 0] = _bgr(15, 255, 255)           # orange vif
        img[0, 1] = _bgr(15, 30, 255)            # orange très délavé (presque blanc)
        img[0, 2] = (128, 128, 128)              # gris : S = 0
        got = [int(x) for x in fn(img, LO, HI)[0] // 255]
        _expect(got == [1, 0, 0],
                f"S doit aussi être testée : orange vif / délavé / gris → attendu [1, 0, 0], vous obtenez {got}")

    def t_input():
        img = two_colors()
        before = img.copy()
        fn(img, LO, HI)
        _expect(np.array_equal(img, before), "l'image d'entrée a été modifiée : travaillez sur le résultat de `cvtColor`, pas sur `img_bgr`")

    return _run([
        ("Renvoie un tableau (H, W) de type uint8", t_form, True),
        ("Ne contient que des 0 et des 255", t_values, True),
        ("Orange gardé, bleu écarté", t_colors, False),
        ("Bornes incluses (H = bas et H = haut passent)", t_bounds, False),
        ("Saturation prise en compte (gris et délavé écartés)", t_saturation, False),
        ("L'image d'entrée n'est pas modifiée", t_input, False),
    ])


# ---------------------------------------------------------------------------
# Exercice : clean_mask(mask, k_open, k_close)
# ---------------------------------------------------------------------------

def _square_mask():
    m = np.zeros((100, 120), np.uint8)
    m[30:70, 40:80] = 255                       # objet 40 × 40
    return m


def check_clean_mask(fn):
    def t_form():
        _is_mask(fn(_square_mask(), 5, 7), (100, 120))

    def t_noise():
        m = _square_mask()
        pts = [(10, 10), (12, 100), (85, 15), (90, 110), (50, 20)]
        for v, u in pts:
            m[v, u] = 255
            m[v + 1, u + 1] = 255               # petits amas de 2 pixels
        out = fn(m, 5, 1)
        left = [p for p in pts if out[p] != 0]
        _expect(not left, f"{len(left)} petit(s) point(s) de bruit subsistent : l'**ouverture** doit les supprimer")

    def t_object():
        out = fn(_square_mask(), 5, 7)
        core = out[34:66, 44:76]
        _expect((core == 255).all(), "l'objet principal est abîmé : l'ouverture ne doit enlever que les petits objets")

    def t_holes():
        m = _square_mask()
        m[48:51, 58:61] = 0                      # trou 3 × 3 dans l'objet
        out = fn(m, 1, 7)
        _expect(out[49, 59] == 255, "le trou dans l'objet n'est pas comblé : la **fermeture** doit le boucher")

    def t_identity():
        rng = np.random.default_rng(1)
        m = (rng.random((50, 60)) > 0.5).astype(np.uint8) * 255
        _expect(np.array_equal(fn(m, 1, 1), m), "avec des motifs de taille 1 (« aucune »), le masque doit rester identique")

    def t_reference():
        rng = np.random.default_rng(0)
        m = np.zeros((120, 160), np.uint8)
        for _ in range(5):
            cv2.circle(m, (int(rng.integers(20, 140)), int(rng.integers(20, 100))), int(rng.integers(8, 20)), 255, -1)
        m[rng.random(m.shape) > 0.97] = 255      # bruit « poivre »
        m[rng.random(m.shape) > 0.985] = 0       # petits trous
        want = vision_ref.clean_mask(m, 7, 15)
        got = fn(m, 7, 15)
        nd = int(np.count_nonzero(got != want))
        _expect(nd == 0, f"{nd} pixels diffèrent de la référence : motif **elliptique** (`MORPH_ELLIPSE`), "
                         f"ouverture **puis** fermeture, avec k_open = 7 et k_close = 15")

    def t_input():
        m = _square_mask()
        before = m.copy()
        fn(m, 5, 7)
        _expect(np.array_equal(m, before), "le masque d'entrée a été modifié")

    return _run([
        ("Renvoie un tableau (H, W) de type uint8", t_form, True),
        ("Supprime le bruit isolé (ouverture)", t_noise, False),
        ("Conserve l'objet principal", t_object, False),
        ("Comble les petits trous (fermeture)", t_holes, False),
        ("Motif de taille 1 = aucune modification", t_identity, False),
        ("Identique à la référence (ellipse, ouverture puis fermeture)", t_reference, False),
        ("Le masque d'entrée n'est pas modifié", t_input, False),
    ])



# ---------------------------------------------------------------------------
# Petits exercices du notebook : un ou deux calculs sur les variables du notebook
# check_snippet(id, sorties_de_l_étudiant, entrées) -> liste de Result
# ---------------------------------------------------------------------------

def _arr(x):
    return isinstance(x, np.ndarray)


def _same(a, b):
    return _arr(a) and _arr(b) and a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b)


def _missing(out, *names):
    for n in names:
        _expect(out.get(n) is not None, f"`{n}` n'a pas de valeur : remplacez `None` par votre code")


def _as_img(out, name, shape, dtype=np.uint8):
    v = out.get(name)
    _expect(v is not None, f"`{name}` n'a pas de valeur : remplacez `None` par votre code")
    _expect(_arr(v), f"`{name}` doit être un tableau NumPy, pas un `{type(v).__name__}`")
    _expect(v.shape == shape, f"`{name}` doit avoir la forme {shape}, la vôtre est {v.shape}")
    _expect(v.dtype == dtype, f"`{name}` doit être de type `{np.dtype(dtype).name}`, le vôtre est `{v.dtype}`")


def _snip_pixel(out, inp):
    img, pu, pv = inp["img"], inp["pu"], inp["pv"]
    want = tuple(int(x) for x in img[pv, pu])

    def t_values():
        _missing(out, "b", "g", "r")
        for n in "bgr":
            try:
                int(out[n])
            except Exception:
                raise AssertionError(f"`{n}` doit être un nombre entier, pas un `{type(out[n]).__name__}`")

    def t_good():
        got = tuple(int(out[n]) for n in "bgr")
        if got == want:
            return
        msg = f"attendu (B, G, R) = {want}, vous obtenez {got}."
        if got == want[::-1]:
            msg += " Les canaux sont à l'envers : OpenCV range **B, G, R**."
        elif 0 <= pu < img.shape[0] and 0 <= pv < img.shape[1] and tuple(int(x) for x in img[pu, pv]) == got:
            msg += " Vous avez lu `[u, v]` : NumPy indexe **`[v, u]`**, la **ligne d'abord**."
        raise AssertionError(msg)

    return _run([("Trois nombres entiers", t_values, True), ("Les bonnes valeurs, dans l'ordre B, G, R", t_good, False)])


def _snip_rgb(out, inp):
    img = inp["img"]
    want = vision_ref._ref_rgb(img)["rgb"]

    def t_form():
        _as_img(out, "rgb", img.shape)

    def t_good():
        got = out["rgb"]
        if np.array_equal(got, img):
            raise AssertionError("l'image n'a pas changé : il faut échanger les canaux B et R")
        _expect(np.array_equal(got, want), "les canaux ne sont pas dans l'ordre R, G, B")

    return _run([("Tableau de même forme et même type que `img`", t_form, True), ("Canaux dans l'ordre R, G, B", t_good, False)])


def _snip_hsv(out, inp):
    img = inp["img"]
    want = vision_ref._ref_hsv(img)["hsv"]

    def t_form():
        _as_img(out, "hsv", img.shape)

    def t_range():
        mx = int(out["hsv"][:, :, 0].max())
        _expect(mx <= 179, f"H va jusqu'à {mx} : en 8 bits OpenCV garde **H ∈ [0, 179]** (la variante « _FULL » donne 0-255)")

    def t_good():
        got = out["hsv"]
        if np.array_equal(got, img):
            raise AssertionError("l'image n'a pas changé : elle n'est pas convertie")
        if np.array_equal(got, cv2.cvtColor(img, cv2.COLOR_RGB2HSV)):
            raise AssertionError("conversion faite depuis **RGB** : une image lue par OpenCV est en **BGR**")
        _expect(np.array_equal(got, want), "ce n'est pas la conversion BGR → HSV attendue")

    return _run([("Tableau (H, W, 3) de type uint8", t_form, True), ("H reste dans [0, 179]", t_range, False),
                 ("Conversion BGR → HSV correcte", t_good, False)])


def _snip_split(out, inp):
    hsv = inp["hsv"]
    want = vision_ref._ref_split(hsv)
    shape = hsv.shape[:2]

    def t_form():
        for n in "HSV":
            _as_img(out, n, shape)

    def t_good():
        got = [out[n] for n in "HSV"]
        w = [want[n] for n in "HSV"]
        if all(np.array_equal(g, x) for g, x in zip(got, w)):
            return
        for perm in ((1, 0, 2), (2, 1, 0), (0, 2, 1), (1, 2, 0), (2, 0, 1)):
            if all(np.array_equal(got[i], w[perm[i]]) for i in range(3)):
                raise AssertionError("les trois canaux sont là mais dans le mauvais ordre : **H, S, V**")
        raise AssertionError("H, S, V doivent être les canaux 0, 1, 2 de `hsv`")

    return _run([("Trois tableaux 2D (H, W) de type uint8", t_form, True), ("H, S, V correspondent aux canaux 0, 1, 2", t_good, False)])


def _snip_mask(out, inp):
    hsv, lo, hi = inp["hsv"], np.asarray(inp["lo"]), np.asarray(inp["hi"])
    want = vision_ref._ref_mask(hsv, lo, hi)["mask"]

    def t_form():
        v = out.get("mask")
        _expect(v is not None, "`mask` n'a pas de valeur : remplacez `None` par votre code")
        _expect(_arr(v), f"`mask` doit être un tableau NumPy, pas un `{type(v).__name__}`")
        if v.ndim == 3:
            raise AssertionError(f"`mask` a la forme {v.shape} : il faut **une** valeur par pixel, (H, W), pas une par canal")
        _expect(v.shape == want.shape, f"`mask` doit avoir la forme {want.shape}, la vôtre est {v.shape}")
        _expect(v.dtype != bool, "`mask` est un tableau de booléens : le masque attendu vaut **0 ou 255** (uint8)")
        _expect(v.dtype == np.uint8, f"`mask` doit être de type `uint8`, le vôtre est `{v.dtype}`")

    def t_values():
        u = set(int(x) for x in np.unique(out["mask"]))
        _expect(u <= {0, 255}, f"le masque ne doit contenir que 0 et 255, il contient aussi {sorted(u - {0, 255})[:5]}")

    def t_good():
        got = out["mask"]
        if np.array_equal(got, want):
            return
        excl = (((hsv > lo) & (hsv < hi)).all(axis=2) * 255).astype(np.uint8)
        if np.array_equal(got, excl):
            raise AssertionError("bornes **exclues** : une valeur égale à `lo` ou à `hi` doit être gardée")
        nd = int(np.count_nonzero(got != want))
        raise AssertionError(f"{nd} pixels diffèrent du masque attendu : testez H, S **et** V entre `lo` et `hi`")

    return _run([("Un tableau (H, W) de type uint8", t_form, True), ("Ne contient que 0 et 255", t_values, True),
                 ("Même masque que celui attendu (bornes incluses)", t_good, False)])


def _snip_objet(out, inp):
    img, mask = inp["img"], inp["mask"]

    def t_form():
        _as_img(out, "objet", img.shape)

    def t_good():
        o = out["objet"]
        keep = mask > 0
        _expect(bool((o[~keep] == 0).all()), "des pixels **hors masque** ne sont pas noirs")
        _expect(bool((o[keep] == img[keep]).all()), "des pixels **dans le masque** ont été modifiés : ils doivent garder leur couleur")

    return _run([("Même forme que `img`", t_form, True), ("Noir hors masque, inchangé dans le masque", t_good, False)])


def _snip_morph(var, ref_fn, other_fn, in_name, k_name, what):
    def f(out, inp):
        inp = {a: b for a, b in inp.items() if a != "_alts"}
        src, k = inp[in_name], int(inp[k_name])
        want = ref_fn(**inp)[var]

        def t_form():
            _as_img(out, var, src.shape)

        def t_good():
            got = out[var]
            if np.array_equal(got, want):
                return
            if np.array_equal(got, src):
                raise AssertionError("aucun effet : le résultat est identique à l'entrée")
            if np.array_equal(got, other_fn(**inp)[var]):
                raise AssertionError(f"c'est l'opération **inverse** : ici il faut {what}")
            rect = cv2.morphologyEx(src, cv2.MORPH_OPEN if var == "opened" else cv2.MORPH_CLOSE,
                                    cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
            if np.array_equal(got, rect):
                raise AssertionError("motif **carré** : l'énoncé demande un motif **elliptique**")
            raise AssertionError(f"résultat différent de l'attendu : {what}, motif elliptique de côté {k}")

        return _run([("Même forme que l'entrée, type uint8", t_form, True), (f"Résultat attendu ({what})", t_good, False)])
    return f


_snip_open = _snip_morph("opened", vision_ref._ref_open,
                         lambda mask, k_open: {"opened": vision_ref._ref_close(mask, k_open)["closed"]},
                         "mask", "k_open", "une ouverture")
_snip_close = _snip_morph("closed", vision_ref._ref_close,
                          lambda opened, k_close: {"closed": vision_ref._ref_open(opened, k_close)["opened"]},
                          "opened", "k_close", "une fermeture")


# ---------------------------------------------------------------------------
# Sections 4 à 7 : petits exercices
# Les tests tournent sur VOS variables, puis (« _alts ») sur quelques cas synthétiques à réponse connue.
# ---------------------------------------------------------------------------

_K_SYN = np.array([[600.0, 0, 320.0], [0, 450.0, 240.0], [0, 0, 1.0]])


def _blobs():
    """Masque de test : trois objets de 20×20, 60×60 et 10×10 (le plus grand est au milieu de la liste)."""
    m = np.zeros((200, 240), np.uint8)
    m[20:40, 20:40] = 255
    m[70:130, 100:160] = 255
    m[150:160, 200:210] = 255
    return m


def _square_moments_mask():
    m = np.zeros((100, 120), np.uint8)
    m[30:70, 40:80] = 255
    return m


def _aruco_scene(aruco_dict, ids=(3,), size=110):
    """Image BGR blanche avec des marqueurs ArUco aux positions connues. Renvoie (image, {id: coin haut-gauche})."""
    img = np.full((300, 420, 3), 255, np.uint8)
    pos = {}
    for k, i in enumerate(ids):
        i = int(min(i, aruco_dict.bytesList.shape[0] - 1))
        x, y = 40 + 200 * k, 60 + 60 * k
        mk = cv2.aruco.generateImageMarker(aruco_dict, i, size)
        img[y:y + size, x:x + size] = cv2.cvtColor(mk, cv2.COLOR_GRAY2BGR)
        pos[i] = (x, y)
    return img, pos


def _syn_pose(L=0.05, rvec=(0.3, -0.2, 0.1), tvec=(0.02, -0.01, 0.4)):
    rvec, tvec = np.array(rvec, np.float64), np.array(tvec, np.float64)
    obj = _marker_object_points(L)
    pts, _ = cv2.projectPoints(obj, rvec, tvec, _K_SYN, None)
    return obj, pts.reshape(4, 2), rvec, tvec


def _ring_mask():
    """Masque de test : un anneau (région avec un trou), un petit objet DANS le trou, et une autre région."""
    m = np.zeros((220, 320), np.uint8)
    cv2.ellipse(m, (110, 110), (90, 80), 0, 0, 360, 255, -1)
    cv2.circle(m, (110, 110), 38, 0, -1)
    cv2.circle(m, (110, 110), 14, 255, -1)
    cv2.rectangle(m, (230, 60), (290, 150), 255, -1)
    return m


def _square_contour():
    cs, _ = cv2.findContours(_square_moments_mask(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cs[0]


def alt_inputs(sid, inp):
    """Entrées supplémentaires (synthétiques) sur lesquelles on réexécute VOTRE code : [(étiquette, entrées), ...]."""
    if sid == "contours":
        return [("anneau, objet dans le trou, autre région", {"mask_clean": _ring_mask()})]
    if sid == "objects":
        cs, _ = cv2.findContours(_blobs(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [("trois objets, seuil 50", {"contours": cs, "min_area": 50}),
                ("deux objets, seuil 200", {"contours": cs, "min_area": 200}),
                ("seuil 5000", {"contours": cs, "min_area": 5000}),
                ("aucun contour", {"contours": (), "min_area": 200})]
    if sid == "moments":
        return [("carré 40 × 40", {"c_obj": _square_contour()})]
    if sid == "centre":
        return [("carré 40 × 40", {"M": cv2.moments(_square_contour())}),
                ("contour d'un seul point", {"M": cv2.moments(np.array([[[5, 5]]], np.int32))})]
    if sid == "objmask":
        return [("carré 40 × 40", {"c_obj": _square_contour(), "mask_clean": _square_moments_mask()})]
    if sid == "zpx":
        d = np.zeros((6, 8), np.uint16)
        d[4, 2] = 1234
        return [("petit tableau", {"depth": d, "depth_scale": 0.001, "pu": 2, "pv": 4})]
    if sid == "zobj":
        d, m = _depth_case()
        return [("objet avec des 0 et des valeurs aberrantes", {"depth": d, "obj_mask": m, "depth_scale": 0.0001})]
    if sid == "xyz":
        return [("fx ≠ fy", {"u_c": 440.0, "v_c": 300.0, "z_obj": 0.5, "K": _K_SYN})]
    if sid == "aruco":
        ad = inp["aruco_dict"]
        img, _ = _aruco_scene(ad, (3,))
        return [("marqueur synthétique", {"img": img, "aruco_dict": ad, "params": inp["params"]})]
    if sid == "obj3d":
        return [("L = 0,10 m", {"marker_len": 0.10})]
    if sid == "pnp":
        obj, pts, _, _ = _syn_pose()
        return [("pose connue", {"obj": obj, "img_pts": pts, "K": _K_SYN, "dist_coeffs": np.zeros(5)})]
    return []


def _alt(inp, label):
    for a in inp.get("_alts", []):
        if a["label"] == label:
            return a
    return None


def _alt_out(inp, label):
    """Sorties de votre code sur le cas synthétique `label` (lève un message clair si votre code y a planté)."""
    a = _alt(inp, label)
    _expect(a is not None, "cas de test indisponible")
    if a["err"]:
        raise AssertionError(f"votre code plante dans un cas particulier ({label}) : {a['err']}")
    return a["out"], a["inputs"]


def _is_contour(c):
    return _arr(c) and c.ndim == 3 and c.shape[1] == 1 and c.shape[2] == 2


def _snip_contours(out, inp):
    mask = inp["mask_clean"]

    def judge(got, mask, label=""):
        want = vision_ref._ref_contours(mask)["contours"]
        _expect(got is not None, f"{label}`contours` n'a pas de valeur : remplacez `None` par votre code")
        if isinstance(got, tuple) and len(got) == 2 and (got[1] is None or (_arr(got[1]) and got[1].ndim == 3 and got[1].shape[2] == 4)):
            raise AssertionError(f"{label}`findContours` renvoie **deux** valeurs `(contours, hierarchy)` : récupérez-les avec `contours, _ = ...`")
        _expect(isinstance(got, (list, tuple)), f"{label}`contours` doit être une liste de contours, pas un `{type(got).__name__}`")
        _expect(all(_is_contour(c) for c in got), f"{label}chaque contour doit être un tableau `(N, 1, 2)` de points (u, v)")
        if len(got) != len(want):
            for name in ("RETR_LIST", "RETR_CCOMP", "RETR_TREE"):
                alt, _ = cv2.findContours(mask, getattr(cv2, name), cv2.CHAIN_APPROX_SIMPLE)
                if len(alt) == len(got):
                    raise AssertionError(f"{label}{len(got)} contours au lieu de {len(want)} : vous avez pris aussi les contours des **trous** et des objets emboîtés "
                                         f"(`{name}`). Il faut les contours **extérieurs** seulement.")
            raise AssertionError(f"{label}{len(got)} contours au lieu de {len(want)} : vérifiez le **mode** de recherche (contours extérieurs seulement)")
        ng, nw = sum(len(c) for c in got), sum(len(c) for c in want)
        if ng > nw:
            raise AssertionError(f"{label}{ng} points au lieu de {nw} : vous gardez **tous** les points de chaque contour. "
                                 "Ici on veut la méthode qui ne garde que les extrémités des segments.")
        _expect(all(np.array_equal(a, b) for a, b in zip(got, want)), f"{label}contours différents de ceux attendus : mode « extérieurs seulement », méthode « extrémités des segments »")

    def t_form():
        v = out.get("contours")
        _expect(v is not None, "`contours` n'a pas de valeur : remplacez `None` par votre code")
        if isinstance(v, tuple) and len(v) == 2 and (v[1] is None or (_arr(v[1]) and v[1].ndim == 3 and v[1].shape[2] == 4)):
            raise AssertionError("`findContours` renvoie **deux** valeurs `(contours, hierarchy)` : récupérez-les avec `contours, _ = ...`")
        _expect(isinstance(v, (list, tuple)), f"`contours` doit être une liste de contours, pas un `{type(v).__name__}`")
        _expect(all(_is_contour(c) for c in v), "chaque contour doit être un tableau `(N, 1, 2)` de points (u, v)")

    def t_real():
        judge(out["contours"], mask)

    def t_ring():
        o, i = _alt_out(inp, "anneau, objet dans le trou, autre région")
        judge(o.get("contours"), i["mask_clean"], "Masque avec un trou : ")

    return _run([("Une liste de contours `(N, 1, 2)`", t_form, True), ("Bons contours sur votre masque", t_real, False),
                 ("Masque avec un trou : seulement les contours extérieurs", t_ring, False)])


def _snip_objects(out, inp):
    contours, min_area = inp["contours"], inp["min_area"]

    def judge(got, contours, min_area, label=""):
        want = vision_ref._ref_objects(contours, min_area)["objects"]
        _expect(got is not None, f"{label}`objects` vaut `None` : s'il n'y a aucun objet, c'est une liste **vide** `[]`")
        if _is_contour(got):
            raise AssertionError(f"{label}`objects` doit être une **liste** de contours (tous ceux qui dépassent le seuil), pas un seul contour")
        _expect(isinstance(got, (list, tuple)), f"{label}`objects` doit être une liste, pas un `{type(got).__name__}`")
        _expect(all(_is_contour(c) for c in got), f"{label}chaque élément doit être un contour `(N, 1, 2)`")
        if len(got) != len(want):
            if len(got) == 1 and len(want) > 1:
                raise AssertionError(f"{label}un seul contour gardé alors que {len(want)} dépassent le seuil : on garde **tous** ceux dont l'aire est ≥ `min_area`")
            raise AssertionError(f"{label}{len(got)} objet(s) au lieu de {len(want)} : on garde les contours dont l'aire (`contourArea`) est ≥ `min_area`")
        areas = [cv2.contourArea(c) for c in got]
        _expect(areas == sorted(areas, reverse=True), f"{label}les objets ne sont pas triés du plus **grand** au plus petit (aires {[round(a) for a in areas]}) : `sorted(..., key=cv2.contourArea, reverse=True)`")
        _expect(all(np.array_equal(a, b) for a, b in zip(got, want)), f"{label}contours différents de ceux attendus")

    def t_real():
        _expect("objects" in out, "`objects` n'est pas défini")
        judge(out["objects"], contours, min_area)

    def alt(label, name):
        def t():
            o, i = _alt_out(inp, label)
            judge(o.get("objects"), i["contours"], i["min_area"], name)
        return t

    return _run([("Résultat attendu sur vos contours", t_real, True),
                 ("Trois objets, seuil 50 : tous gardés, du plus grand au plus petit", alt("trois objets, seuil 50", "Trois objets : "), False),
                 ("Deux objets, seuil 200 : le plus petit est écarté", alt("deux objets, seuil 200", "Seuil 200 : "), False),
                 ("Seuil trop haut : liste vide", alt("seuil 5000", "Seuil 5000 : "), False),
                 ("Aucun contour : liste vide", alt("aucun contour", "Liste vide : "), False)])


def _snip_moments(out, inp):
    c = inp["c_obj"]
    want = vision_ref._ref_moments(c)["M"]

    def t_form():
        M = out.get("M")
        _expect(M is not None, "`M` n'a pas de valeur : remplacez `None` par votre code")
        _expect(isinstance(M, dict) and all(k in M for k in ("m00", "m10", "m01")),
                "`M` doit être le dictionnaire renvoyé par `cv2.moments` (clés `m00`, `m10`, `m01`…)")

    def t_good():
        M = out["M"]
        _expect(abs(M["m00"] - want["m00"]) < 1e-6 and abs(M["m10"] - want["m10"]) < 1e-3,
                f"`m00 = {M['m00']:.0f}` mais l'aire du contour `c_obj` est {want['m00']:.0f} : calculez les moments du **contour** `c_obj`")

    def t_square():
        o, i = _alt_out(inp, "carré 40 × 40")
        m00 = o["M"]["m00"] if isinstance(o.get("M"), dict) else None
        _expect(m00 == 1521, f"sur le contour d'un carré de 40 × 40 pixels, `m00` doit valoir 1521 (l'aire du contour, 39 × 39), pas {m00}")

    return _run([("`M` est un dictionnaire de moments", t_form, True), ("`m00` = aire du contour de l'objet", t_good, False),
                 ("Carré 40 × 40 : m00 = 1521", t_square, False)])


def _snip_centre(out, inp):
    M = inp["M"]
    want = vision_ref._ref_centre(M)

    def num(v):
        try:
            return float(v)
        except Exception:
            return None

    def t_form():
        _missing(out, "u_c", "v_c")
        _expect(num(out["u_c"]) is not None and num(out["v_c"]) is not None, "`u_c` et `v_c` doivent être des nombres")

    def t_good():
        u, v = num(out["u_c"]), num(out["v_c"])
        if abs(u - want["u_c"]) < 1e-6 and abs(v - want["v_c"]) < 1e-6:
            return
        if abs(u - want["v_c"]) < 1e-6 and abs(v - want["u_c"]) < 1e-6:
            raise AssertionError("`u_c` et `v_c` sont inversés : `u_c = m10 / m00` (la **colonne**), `v_c = m01 / m00` (la **ligne**)")
        raise AssertionError(f"attendu (u_c, v_c) = ({want['u_c']:.2f}, {want['v_c']:.2f}), vous obtenez ({u:.2f}, {v:.2f}) : `m10 / m00` et `m01 / m00`")

    def t_square():
        o, _i = _alt_out(inp, "carré 40 × 40")
        u, v = num(o.get("u_c")), num(o.get("v_c"))
        _expect(u is not None and abs(u - 59.5) < 1e-6 and abs(v - 49.5) < 1e-6,
                f"carré occupant les colonnes 40 à 79 et les lignes 30 à 69 : le centre est (59,5 ; 49,5), vous obtenez ({u}, {v})")

    def t_point():
        o, _i = _alt_out(inp, "contour d'un seul point")
        _expect(o.get("u_c") is None and o.get("v_c") is None,
                "contour réduit à un point (aire nulle, `m00 = 0`) : pas de centroïde, `u_c` et `v_c` doivent rester `None`. Testez `m00 > 0` avant de diviser.")

    return _run([("Deux nombres `u_c` et `v_c`", t_form, True), ("Bon centroïde sur votre objet", t_good, False),
                 ("Carré 40 × 40 : (59,5 ; 49,5)", t_square, False), ("Contour d'un seul point : pas de division par zéro", t_point, False)])


def _snip_objmask(out, inp):
    c, mc = inp["c_obj"], inp["mask_clean"]

    def judge(o, c, shape, label=""):
        _as_img(o, "obj_mask", shape)
        got = o["obj_mask"]
        want = _contour_mask(c, shape)
        u = set(int(x) for x in np.unique(got))
        if u <= {0, 1} and u != {0}:
            raise AssertionError(f"{label}le masque vaut 1 : un masque se note **255** (blanc) / 0 (noir)")
        _expect(u <= {0, 255}, f"{label}le masque ne doit contenir que 0 et 255")
        if np.array_equal(got, want):
            return
        if np.count_nonzero(got) < np.count_nonzero(want) and not (got[want == 0] > 0).any():
            raise AssertionError(f"{label}seul le **trait** du contour est blanc : il faut remplir tout l'**intérieur** (épaisseur `cv2.FILLED`)")
        raise AssertionError(f"{label}{int(np.count_nonzero(got != want))} pixels diffèrent du masque attendu : blanc (255) sur tout l'intérieur du contour, noir ailleurs")

    def t_form():
        _as_img(out, "obj_mask", mc.shape)

    def t_real():
        judge(out, c, mc.shape)

    def t_square():
        o, i = _alt_out(inp, "carré 40 × 40")
        judge(o, i["c_obj"], i["mask_clean"].shape, "Carré 40 × 40 : ")

    return _run([("Tableau (H, W) de type uint8", t_form, True), ("Intérieur du contour blanc (255), reste noir", t_real, False),
                 ("Carré 40 × 40 : 1600 pixels blancs", t_square, False)])


def _snip_zpx(out, inp):
    depth, scale, pu, pv = inp["depth"], inp["depth_scale"], inp["pu"], inp["pv"]
    want = vision_ref._ref_zpx(depth, scale, pu, pv)["z_px"]

    def num(v):
        try:
            return float(v)
        except Exception:
            return None

    def judge(z, depth, scale, pu, pv, label=""):
        _expect(z is not None, f"{label}`z_px` n'est pas un nombre")
        want = float(depth[pv, pu]) * scale
        if abs(z - want) < 1e-9:
            return
        raw = float(depth[pv, pu])
        if raw != 0 and abs(z - raw) < 1e-9:
            raise AssertionError(f"{label}`z_px = {z:g}` est la valeur **brute** : multipliez par `depth_scale` pour obtenir des **mètres**")
        if 0 <= pu < depth.shape[0] and 0 <= pv < depth.shape[1] and abs(z - float(depth[pu, pv]) * scale) < 1e-9:
            raise AssertionError(f"{label}vous avez lu `depth[u, v]` : NumPy indexe **`[v, u]`**, la ligne d'abord")
        raise AssertionError(f"{label}attendu {want:.4f} m, vous obtenez {z:.4f}")

    def t_form():
        _missing(out, "z_px")
        _expect(num(out["z_px"]) is not None, f"`z_px` doit être un nombre, pas un `{type(out['z_px']).__name__}`")

    def t_good():
        judge(num(out["z_px"]), depth, scale, pu, pv)

    def t_small():
        o, i = _alt_out(inp, "petit tableau")
        judge(num(o.get("z_px")), i["depth"], i["depth_scale"], i["pu"], i["pv"], "Petit tableau : ")

    return _run([("Un nombre", t_form, True), ("Profondeur en mètres au pixel choisi", t_good, False),
                 ("Petit tableau de test : [v, u] et unité", t_small, False)])


def _snip_zobj(out, inp):
    depth, mask, scale = inp["depth"], inp["obj_mask"], inp["depth_scale"]

    def num(v):
        try:
            x = float(v)
            return x
        except Exception:
            return None

    def judge(z, depth, mask, scale, label=""):
        _expect(z is not None, f"{label}`z_obj` doit être un nombre")
        _expect(not np.isnan(z), f"{label}`z_obj` vaut `nan` : aucun pixel sélectionné ? Le masque vaut **255**, pas 1 ; utilisez `mask > 0`")
        valid = (mask > 0) & (depth > 0)
        want = float(np.median(depth[valid])) * scale
        if abs(z - want) < 1e-6:
            return
        allv = depth[mask > 0].astype(np.float64)
        if abs(z - allv.mean() * scale) < 1e-6 and (depth[mask > 0] == 0).any():
            raise AssertionError(f"{label}la moyenne **avec les 0** (pixels sans mesure) tire la distance vers le bas : ne gardez que `depth > 0`")
        if abs(z - float(np.median(allv)) * scale) < 1e-6:
            raise AssertionError(f"{label}médiane faite **avec les 0** : sélectionnez d'abord les pixels dont `depth > 0`")
        if abs(z - float(depth[valid].astype(np.float64).mean()) * scale) < 1e-6:
            raise AssertionError(f"{label}la **moyenne** est tirée par les valeurs aberrantes (bords, fond) : utilisez la **médiane**")
        if abs(z - float(np.median(depth[valid]))) < 1e-6:
            raise AssertionError(f"{label}valeur **brute** : multipliez par `depth_scale` pour avoir des mètres")
        raise AssertionError(f"{label}attendu {want:.4f} m, vous obtenez {z:.4f} m : médiane des pixels **du masque** dont la mesure est valide")

    def t_form():
        _missing(out, "z_obj")
        _expect(num(out["z_obj"]) is not None, f"`z_obj` doit être un nombre, pas un `{type(out['z_obj']).__name__}`")

    def t_good():
        judge(num(out["z_obj"]), depth, mask, scale)

    def t_syn():
        o, i = _alt_out(inp, "objet avec des 0 et des valeurs aberrantes")
        judge(num(o.get("z_obj")), i["depth"], i["obj_mask"], i["depth_scale"], "Cas test : ")

    return _run([("Un nombre", t_form, True), ("Bonne profondeur sur votre objet", t_good, False),
                 ("Cas test avec des 0 et des valeurs aberrantes", t_syn, False)])


def _snip_xyz(out, inp):
    u, v, z, K = inp["u_c"], inp["v_c"], inp["z_obj"], inp["K"]

    def judge(o, u, v, z, K, label=""):
        got = []
        for n in "XYZ":
            _expect(o.get(n) is not None, f"{label}`{n}` n'a pas de valeur")
            try:
                got.append(float(o[n]))
            except Exception:
                raise AssertionError(f"{label}`{n}` doit être un nombre")
        want = vision_ref.deproject(u, v, z, K)
        if np.allclose(got, want, atol=1e-9):
            return
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        if abs(got[2] - z) > 1e-9:
            raise AssertionError(f"{label}`Z` doit rester égal à la profondeur mesurée `z_obj`")
        if abs(got[0] - z * u / fx) < 1e-9:
            raise AssertionError(f"{label}`X` : il faut soustraire le point principal, `(u - cx)`")
        if abs(got[1] - z * v / fy) < 1e-9:
            raise AssertionError(f"{label}`Y` : il faut soustraire le point principal, `(v - cy)`")
        if abs(got[0] - z * (u - cx) / fy) < 1e-9 or abs(got[1] - z * (v - cy) / fx) < 1e-9:
            raise AssertionError(f"{label}`X` utilise **fx** et `Y` utilise **fy**")
        raise AssertionError(f"{label}attendu X = {want[0]:.4f}, Y = {want[1]:.4f} ; vous obtenez X = {got[0]:.4f}, Y = {got[1]:.4f} : X = Z·(u − cx)/fx, Y = Z·(v − cy)/fy")

    def t_real():
        judge(out, u, v, z, K)

    def t_syn():
        o, i = _alt_out(inp, "fx ≠ fy")
        judge(o, i["u_c"], i["v_c"], i["z_obj"], i["K"], "Cas test (fx ≠ fy) : ")

    return _run([("Bons X, Y, Z sur votre objet", t_real, True), ("Cas test avec fx ≠ fy", t_syn, False)])


def _snip_aruco(out, inp):
    img, ad, params = inp["img"], inp["aruco_dict"], inp["params"]
    want = vision_ref._ref_aruco(img, ad, params)

    def t_gray():
        g = out.get("gray")
        _expect(g is not None, "`gray` n'a pas de valeur : remplacez `None` par votre code")
        _expect(_arr(g) and g.shape == img.shape[:2], f"`gray` doit être une image à **un seul canal**, de forme {img.shape[:2]}")
        _expect(np.array_equal(g, want["gray"]), "ce n'est pas la conversion BGR → niveaux de gris : `cv2.COLOR_BGR2GRAY`")

    def judge(o, w, label=""):
        for n in ("corners", "ids", "rejected"):
            _expect(n in o, f"{label}`{n}` n'est pas défini")
        c, i = o["corners"], o["ids"]
        if i is not None and not _arr(i) and _arr(c):
            raise AssertionError(f"{label}`corners` et `ids` sont inversés : `detectMarkers` renvoie `corners, ids, rejected` dans cet ordre")
        wi = w["ids"]
        if wi is None:
            _expect(i is None, f"{label}aucun marqueur attendu ici")
            return
        _expect(i is not None, f"{label}`ids` vaut `None` : aucun marqueur détecté. Avez-vous passé l'image `gray` à `detectMarkers`, avec le même dictionnaire `aruco_dict` ?")
        _expect(_arr(i) and np.array_equal(i, wi), f"{label}ids détectés {i.ravel().tolist()} au lieu de {wi.ravel().tolist()}")
        _expect(len(c) == len(w["corners"]) and all(np.allclose(a, b) for a, b in zip(c, w["corners"])), f"{label}les coins ne correspondent pas à ceux attendus")

    def t_real():
        judge(out, want)

    def t_syn():
        o, i = _alt_out(inp, "marqueur synthétique")
        judge(o, vision_ref._ref_aruco(i["img"], i["aruco_dict"], i["params"]), "Image test : ")
        _expect(o["ids"] is not None, "Image test : le marqueur n'est pas détecté")

    return _run([("`gray` : image à un canal, en niveaux de gris", t_gray, True), ("`corners`, `ids` : mêmes résultats que l'attendu", t_real, False),
                 ("Détecte un marqueur synthétique connu", t_syn, False)])


def _snip_obj3d(out, inp):
    L = float(inp["marker_len"])

    def judge(o, L, label=""):
        v = o.get("obj")
        _expect(v is not None, f"{label}`obj` n'a pas de valeur : remplacez `None` par votre code")
        _expect(_arr(v) and v.shape == (4, 3), f"{label}`obj` doit être un tableau de forme (4, 3) : 4 coins, 3 coordonnées (x, y, z)")
        _expect(np.issubdtype(v.dtype, np.floating), f"{label}`obj` doit contenir des nombres décimaux (float32), pas des `{v.dtype}`")
        want = _marker_object_points(L)
        if np.allclose(v, want, atol=1e-6):
            return
        _expect(np.allclose(v[:, 2], 0), f"{label}le marqueur est **plan** : la 3ᵉ coordonnée (z) de chaque coin vaut 0")
        if np.allclose(np.abs(v[:, :2]), L, atol=1e-6):
            raise AssertionError(f"{label}les coins sont à **±L/2** du centre (le centre est l'origine), pas à ±L")
        if sorted(map(tuple, np.round(v, 6))) == sorted(map(tuple, np.round(want, 6))):
            raise AssertionError(f"{label}bons coins, mauvais **ordre** : haut-gauche (−L/2, +L/2), haut-droit (+L/2, +L/2), bas-droit (+L/2, −L/2), bas-gauche (−L/2, −L/2)")
        raise AssertionError(f"{label}coins différents de l'attendu : carré de côté L centré sur l'origine, dans le plan z = 0")

    def t_real():
        judge(out, L)

    def t_syn():
        o, i = _alt_out(inp, "L = 0,10 m")
        judge(o, i["marker_len"], "Avec L = 0,10 m : ")

    return _run([("Tableau (4, 3) décimal, bon ordre et bonnes valeurs", t_real, True), ("Avec L = 0,10 m aussi (utilise bien `marker_len`)", t_syn, False)])


def _snip_pnp(out, inp):
    obj, pts, K, dist = inp["obj"], inp["img_pts"], inp["K"], inp["dist_coeffs"]

    def judge(o, obj, pts, K, dist, label=""):
        for n in ("ok", "rvec", "tvec"):
            _expect(o.get(n) is not None, f"{label}`{n}` n'a pas de valeur : `ok, rvec, tvec = cv2.solvePnP(...)`")
        _expect(bool(o["ok"]), f"{label}`solvePnP` a échoué (`ok = False`)")
        rv, tv = np.asarray(o["rvec"], np.float64).reshape(-1), np.asarray(o["tvec"], np.float64).reshape(-1)
        _expect(rv.size == 3 and tv.size == 3, f"{label}`rvec` et `tvec` doivent avoir 3 composantes")
        wk = vision_ref._ref_pnp(obj, pts, K, dist)
        wt = np.asarray(wk["tvec"]).reshape(-1)
        proj, _ = cv2.projectPoints(obj, rv, tv, K, dist)
        err = float(np.linalg.norm(proj.reshape(4, 2) - pts.reshape(4, 2), axis=1).mean())
        _expect(err < 1.0, f"{label}la pose ne recolle pas aux coins détectés (erreur de reprojection {err:.1f} px) : ordre des arguments `(objet 3D, points image, K, dist)` ?")
        _expect(np.linalg.norm(tv - wt) < 5e-3, f"{label}position trop éloignée de l'attendue ({np.round(tv * 100, 1).tolist()} cm au lieu de {np.round(wt * 100, 1).tolist()} cm)")

    def t_real():
        judge(out, obj, pts, K, dist)

    def t_syn():
        o, i = _alt_out(inp, "pose connue")
        judge(o, i["obj"], i["img_pts"], i["K"], i["dist_coeffs"], "Pose connue : ")
        tv = np.asarray(o["tvec"], np.float64).reshape(-1)
        _expect(np.linalg.norm(tv - np.array([0.02, -0.01, 0.4])) < 2e-3, f"pose synthétique : t devrait valoir (2 ; −1 ; 40) cm, vous obtenez {np.round(tv * 100, 1).tolist()} cm")

    return _run([("`ok, rvec, tvec` cohérents avec les coins détectés", t_real, True), ("Retrouve une pose synthétique connue", t_syn, False)])


# ---------------------------------------------------------------------------
# Fonctions des sections 4 à 7
# ---------------------------------------------------------------------------

def check_find_objects(fn):
    def lst(r):
        _expect(r is not None, "renvoie `None` : s'il n'y a aucun objet, renvoyez une liste **vide** `[]`")
        if _is_contour(r):
            raise AssertionError("un seul contour renvoyé : il faut une **liste** de contours (tous ceux qui dépassent le seuil)")
        _expect(isinstance(r, (list, tuple)), f"il faut renvoyer une liste de contours, pas un `{type(r).__name__}`")
        _expect(all(_is_contour(c) for c in r), "chaque élément doit être un contour : tableau `(N, 1, 2)` de points (u, v)")
        return list(r)

    def t_form():
        lst(fn(_blobs(), 100))

    def t_count():
        r = lst(fn(_blobs(), 100))
        if len(r) == 1:
            raise AssertionError("un seul objet gardé, mais deux dépassent 100 px² (≈ 3500 et ≈ 360) : gardez **tous** les contours d'aire ≥ `min_area`")
        _expect(len(r) == 2, f"avec un seuil de 100 px², 2 objets sont attendus (le 3ᵉ ne fait que 81 px²), la liste en contient {len(r)}")

    def t_sorted():
        r = lst(fn(_blobs(), 50))
        areas = [cv2.contourArea(c) for c in r]
        _expect(len(r) == 3, f"3 objets dépassent 50 px², la liste en contient {len(r)}")
        _expect(areas == sorted(areas, reverse=True), f"la liste doit aller du plus **grand** au plus petit (aires {[round(a) for a in areas]}) : `sorted(..., key=cv2.contourArea, reverse=True)`")

    def t_thresh():
        _expect(len(lst(fn(_blobs(), 5000))) == 0, "aucun objet ne dépasse 5000 px² : renvoyez une liste vide `[]`")
        _expect(len(lst(fn(_blobs(), 300))) == 2, "avec un seuil de 300 px², 2 objets sont attendus (≈ 3500 et ≈ 360)")

    def t_empty():
        _expect(len(lst(fn(np.zeros((60, 80), np.uint8), 100))) == 0, "masque vide : aucun objet, renvoyez une liste vide `[]`")

    def t_ring():
        r = lst(fn(_ring_mask(), 50))
        _expect(len(r) == 2, f"anneau + autre région : 2 objets **extérieurs** attendus (ni le trou ni l'objet dans le trou), la liste en contient {len(r)}")

    def t_ref():
        rng = np.random.default_rng(5)
        m = np.zeros((120, 160), np.uint8)
        for _ in range(6):
            cv2.circle(m, (int(rng.integers(15, 145)), int(rng.integers(15, 105))), int(rng.integers(5, 22)), 255, -1)
        for a in (0, 300):
            got, want = lst(fn(m, a)), vision_ref.find_objects(m, a)
            _expect(len(got) == len(want) and all(np.array_equal(x, y) for x, y in zip(got, want)), f"résultat différent de la référence (seuil {a})")

    def t_input():
        m = _blobs()
        b = m.copy()
        fn(m, 100)
        _expect(np.array_equal(m, b), "le masque d'entrée a été modifié")

    return _run([("Renvoie une liste de contours `(N, 1, 2)`", t_form, True), ("Garde tous les contours au-dessus du seuil", t_count, False),
                 ("Du plus grand au plus petit", t_sorted, False), ("Seuil trop haut : liste vide", t_thresh, False),
                 ("Masque vide : liste vide", t_empty, False), ("Contours extérieurs seulement", t_ring, False),
                 ("Identique à la référence", t_ref, False), ("Le masque d'entrée n'est pas modifié", t_input, False)])


def check_centroid(fn):
    sq = _square_contour()

    def pair(r):
        _expect(r is not None, "renvoie `None` alors que le contour a une aire")
        _expect(len(r) == 2, "il faut renvoyer deux nombres `(u_c, v_c)`")
        return float(r[0]), float(r[1])

    def t_form():
        pair(fn(sq))

    def t_square():
        u, v = pair(fn(sq))
        if abs(u - 49.5) < 1e-6 and abs(v - 59.5) < 1e-6:
            raise AssertionError("`(u_c, v_c)` inversé : `u_c` = la **colonne** (m10 / m00), `v_c` = la **ligne** (m01 / m00)")
        if abs(u - 60) < 1e-6 and abs(v - 50) < 1e-6:
            raise AssertionError("c'est le centre de la **boîte englobante** : le centroïde se calcule avec les **moments** du contour")
        _expect(abs(u - 59.5) < 1e-6 and abs(v - 49.5) < 1e-6, f"carré sur les colonnes 40-79 et les lignes 30-69 : attendu (59,5 ; 49,5), obtenu ({u:.2f} ; {v:.2f})")

    def t_point():
        pt = np.array([[[5, 5]]], np.int32)
        try:
            r = fn(pt)
        except ZeroDivisionError:
            raise AssertionError("division par zéro sur un contour d'aire nulle (un point) : testez `m00 > 0` avant de diviser, sinon renvoyez `None`")
        _expect(r is None, "contour d'aire nulle : pas de centroïde, renvoyez `None`")

    def t_lshape():
        m = np.zeros((100, 120), np.uint8)
        m[20:80, 20:40] = 255
        m[60:80, 40:100] = 255              # forme en L
        c = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]
        u, v = pair(fn(c))
        wu, wv = vision_ref.centroid(c)
        x, y, w, h = cv2.boundingRect(c)
        if min(abs(u - (x + w / 2)), abs(u - (x + (w - 1) / 2))) < 1e-6 and min(abs(v - (y + h / 2)), abs(v - (y + (h - 1) / 2))) < 1e-6:
            raise AssertionError("c'est le centre de la **boîte englobante** : le centroïde se calcule avec les **moments** du contour")
        _expect(abs(u - wu) < 1e-6 and abs(v - wv) < 1e-6, f"forme en L : attendu ({wu:.2f} ; {wv:.2f}), obtenu ({u:.2f} ; {v:.2f})")

    def t_input():
        c = sq.copy()
        fn(c)
        _expect(np.array_equal(c, sq), "le contour d'entrée a été modifié")

    return _run([("Renvoie deux nombres `(u_c, v_c)`", t_form, True), ("Carré : (59,5 ; 49,5), dans l'ordre (u, v)", t_square, False),
                 ("Contour d'un point : renvoie None", t_point, False), ("Forme en L : vrai centroïde", t_lshape, False),
                 ("Le contour d'entrée n'est pas modifié", t_input, False)])


def _depth_case():
    rng = np.random.default_rng(3)
    depth = np.full((100, 120), 1000, np.uint16)
    mask = np.zeros((100, 120), np.uint8)
    mask[20:80, 30:90] = 255
    patch = np.full((60, 60), 3000, np.uint16)
    r = rng.random((60, 60))
    patch[r < 0.2] = 0
    patch[r > 0.9] = 9000
    patch[0, :] = patch[-1, :] = 5000        # pixels de bord : mélange objet / fond
    patch[:, 0] = patch[:, -1] = 5000
    depth[20:80, 30:90] = patch
    return depth, mask


def check_object_depth(fn):
    def cnt():
        d, m = _depth_case()
        return d, cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]

    def num(r):
        _expect(r is not None, "renvoie `None` alors que l'objet a des pixels mesurés")
        try:
            return float(r)
        except Exception:
            raise AssertionError(f"il faut renvoyer un nombre (mètres), pas un `{type(r).__name__}`")

    def t_form():
        d, c = cnt()
        num(fn(d, c, 0.0001))

    def t_value():
        d, c = cnt()
        m = _contour_mask(c, d.shape)
        z = num(fn(d, c, 0.0001))
        valid = (m > 0) & (d > 0)
        want = float(np.median(d[valid])) * 0.0001
        if abs(z - want) < 1e-9:
            return
        if np.isnan(z):
            raise AssertionError("résultat `nan` : aucun pixel sélectionné ? dessinez le contour **rempli** (`cv2.FILLED`) pour obtenir le masque de l'objet")
        if abs(z - 0.5) < 1e-9:
            raise AssertionError("seul le **trait** du contour est utilisé : il faut tous les pixels de l'**intérieur** (contour rempli, `cv2.FILLED`)")
        if abs(z - float(d[m > 0].mean()) * 0.0001) < 1e-9:
            raise AssertionError("moyenne **avec les 0** (pixels sans mesure) : écartez d'abord les pixels `depth == 0`")
        if abs(z - float(np.median(d[m > 0])) * 0.0001) < 1e-9:
            raise AssertionError("médiane faite **avec les 0** : écartez d'abord les pixels `depth == 0`")
        if abs(z - float(d[valid].mean()) * 0.0001) < 1e-9:
            raise AssertionError("la **moyenne** est tirée par les valeurs aberrantes : utilisez la **médiane**")
        if abs(z - float(np.median(d[valid]))) < 1e-9:
            raise AssertionError("valeur **brute** : multipliez par `depth_scale` pour obtenir des mètres")
        raise AssertionError(f"attendu {want:.4f} m, obtenu {z:.4f} m : médiane des pixels **à l'intérieur du contour** dont la mesure est valide")

    def t_scale():
        d, c = cnt()
        a, b = num(fn(d, c, 0.0001)), num(fn(d, c, 0.001))
        _expect(abs(b - 10 * a) < 1e-9, "le résultat doit être proportionnel à `depth_scale`")

    def t_none():
        d, c = cnt()
        d[:] = 0
        try:
            r = fn(d, c, 0.0001)
        except Exception as e:
            raise AssertionError(f"plante quand aucun pixel n'a de mesure ({type(e).__name__}) : renvoyez `None`")
        _expect(r is None, "aucune mesure valide dans l'objet : renvoyez `None` (et non `nan` ou 0)")

    def t_input():
        d, c = cnt()
        d0, c0 = d.copy(), c.copy()
        fn(d, c, 0.0001)
        _expect(np.array_equal(d, d0) and np.array_equal(c, c0), "les tableaux d'entrée ont été modifiés")

    return _run([("Renvoie un nombre", t_form, True), ("Médiane des pixels valides de l'objet, en mètres", t_value, False),
                 ("Proportionnel à `depth_scale`", t_scale, False), ("Aucune mesure valide : renvoie None", t_none, False),
                 ("Les tableaux d'entrée ne sont pas modifiés", t_input, False)])


def check_deproject(fn):
    K1 = np.array([[600.0, 0, 320.0], [0, 600.0, 240.0], [0, 0, 1.0]])

    def p3(r):
        _expect(r is not None, "la fonction renvoie `None` : pensez au `return`")
        a = np.asarray(r, np.float64).reshape(-1)
        _expect(a.size == 3, "il faut renvoyer 3 valeurs (X, Y, Z)")
        return a

    def t_form():
        p3(fn(320, 240, 0.5, K1))

    def t_axis():
        _expect(np.allclose(p3(fn(320, 240, 0.5, K1)), [0, 0, 0.5], atol=1e-9),
                "le pixel du **centre optique** (cx, cy) est sur l'axe : X = Y = 0 ; Z = la profondeur donnée")

    def t_values():
        got = p3(fn(440, 300, 0.5, K1))
        _expect(np.allclose(got, [0.1, 0.05, 0.5], atol=1e-9), f"pixel (440, 300) à 0,5 m, f = 600 : attendu (0,10 ; 0,05 ; 0,50) m, obtenu {np.round(got, 4).tolist()}")

    def t_fxfy():
        got = p3(fn(440, 300, 0.5, _K_SYN))
        want = vision_ref.deproject(440, 300, 0.5, _K_SYN)
        if not np.allclose(got, want, atol=1e-9):
            raise AssertionError("avec fx ≠ fy : **X** utilise `K[0, 0]` (fx) et **Y** utilise `K[1, 1]` (fy) ; le point principal est `K[0, 2]`, `K[1, 2]`")

    def t_roundtrip():
        P = np.array([[0.05, -0.03, 0.35]])
        uv, _ = cv2.projectPoints(P, np.zeros(3), np.zeros(3), _K_SYN, None)
        u, v = uv.reshape(2)
        _expect(np.allclose(p3(fn(u, v, 0.35, _K_SYN)), P[0], atol=1e-6), "projeter un point 3D puis le rétro-projeter doit redonner le même point")

    return _run([("Renvoie 3 valeurs", t_form, True), ("Pixel central : (0, 0, Z)", t_axis, False), ("Valeurs connues (f = 600)", t_values, False),
                 ("fx ≠ fy", t_fxfy, False), ("Aller-retour avec `cv2.projectPoints`", t_roundtrip, False)])


def check_detect_aruco(fn):
    AD = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    DID = cv2.aruco.DICT_4X4_50

    def t_form():
        img, _ = _aruco_scene(AD, (7, 12))
        r = fn(img, DID)
        _expect(r is not None, "la fonction renvoie `None` : pensez au `return`")
        _expect(isinstance(r, (list, tuple)), f"il faut renvoyer une **liste** de (id, coins), pas un `{type(r).__name__}`")
        _expect(len(r) == 2, f"2 marqueurs sont présents, la liste en contient {len(r)}")
        for it in r:
            _expect(len(it) == 2, "chaque élément doit être un couple `(id, coins)`")
            _expect(np.asarray(it[1]).shape == (4, 2), f"les coins doivent former un tableau (4, 2), pas {np.asarray(it[1]).shape} : `corners.reshape(4, 2)`")

    def t_ids():
        img, _ = _aruco_scene(AD, (7, 12))
        got = sorted(int(i) for i, _ in fn(img, DID))
        _expect(got == [7, 12], f"ids attendus [7, 12], obtenus {got}")

    def t_corners():
        img, pos = _aruco_scene(AD, (7, 12))
        for i, c in fn(img, DID):
            c = np.asarray(c, np.float64)
            x, y = pos[int(i)]
            want = np.array([[x, y], [x + 109, y], [x + 109, y + 109], [x, y + 109]], np.float64)
            if np.abs(c - want).max() > 3:
                raise AssertionError(f"coins du marqueur {int(i)} : attendus ≈ {want.astype(int).tolist()} (haut-gauche, haut-droit, bas-droit, bas-gauche), obtenus {np.round(c).astype(int).tolist()}")

    def t_none():
        blank = np.full((200, 300, 3), 255, np.uint8)
        try:
            r = fn(blank, DID)
        except Exception as e:
            raise AssertionError(f"plante quand il n'y a aucun marqueur ({type(e).__name__} : {e}) : `ids` vaut alors `None`, testez-le")
        _expect(r is not None and len(r) == 0, "aucun marqueur : renvoyez une liste **vide** `[]`")

    def t_dict():
        img, _ = _aruco_scene(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50), (4,))
        _expect(len(fn(img, DID)) == 0, "un marqueur 5×5 ne doit pas être reconnu avec le dictionnaire 4×4 : utilisez le paramètre `dict_id`")
        _expect(len(fn(img, cv2.aruco.DICT_5X5_50)) == 1, "avec `DICT_5X5_50`, le marqueur 5×5 doit être reconnu : utilisez le paramètre `dict_id`")

    def t_input():
        img, _ = _aruco_scene(AD, (7,))
        b = img.copy()
        fn(img, DID)
        _expect(np.array_equal(img, b), "l'image d'entrée a été modifiée")

    return _run([("Renvoie une liste de couples (id, coins (4, 2))", t_form, True), ("Bons identifiants", t_ids, False),
                 ("Coins dans l'ordre haut-gauche → bas-gauche", t_corners, False), ("Aucun marqueur : liste vide", t_none, False),
                 ("Le dictionnaire passé en paramètre est utilisé", t_dict, False), ("L'image d'entrée n'est pas modifiée", t_input, False)])


def check_marker_pose(fn):
    obj, pts, rv0, tv0 = _syn_pose()

    def rt(r):
        _expect(r is not None, "renvoie `None` : `solvePnP` a échoué ou le `return` manque")
        _expect(len(r) == 2, "il faut renvoyer le couple `(rvec, tvec)`")
        rv, tv = np.asarray(r[0], np.float64).reshape(-1), np.asarray(r[1], np.float64).reshape(-1)
        _expect(rv.size == 3 and tv.size == 3, "`rvec` et `tvec` ont 3 composantes chacun")
        return rv, tv

    def t_form():
        rt(fn(pts, 0.05, _K_SYN, None))

    def t_tvec():
        _, tv = rt(fn(pts, 0.05, _K_SYN, None))
        _expect(np.linalg.norm(tv - tv0) < 1e-3, f"position attendue (2 ; −1 ; 40) cm, obtenue {np.round(tv * 100, 1).tolist()} cm : coins de l'objet 3D à ±L/2 dans l'ordre haut-gauche → bas-gauche, `flags=cv2.SOLVEPNP_IPPE_SQUARE`")

    def t_reproj():
        rv, tv = rt(fn(pts, 0.05, _K_SYN, None))
        pr, _ = cv2.projectPoints(obj, rv, tv, _K_SYN, None)
        err = float(np.linalg.norm(pr.reshape(4, 2) - pts, axis=1).mean())
        _expect(err < 0.2, f"erreur de reprojection {err:.2f} px : la pose ne recolle pas aux coins")

    def t_L():
        _, tv = rt(fn(pts, 0.10, _K_SYN, None))
        _expect(abs(tv[2] - 0.8) < 3e-3, f"avec un marqueur **deux fois plus grand** vu pareil, il est deux fois plus loin (Z = 0,80 m), vous obtenez {tv[2]:.3f} m : utilisez bien `L`")

    def t_dist():
        _, tv = rt(fn(pts, 0.05, _K_SYN, np.zeros(5)))
        _expect(np.linalg.norm(tv - tv0) < 1e-3, "avec `dist = np.zeros(5)` le résultat doit rester le même")

    def t_shape():
        _, tv = rt(fn(pts.reshape(1, 4, 2).astype(np.float32), 0.05, _K_SYN, None))
        _expect(np.linalg.norm(tv - tv0) < 2e-3, "les coins peuvent arriver sous la forme `(1, 4, 2)` (celle de `detectMarkers`) : `reshape(4, 2)`")

    return _run([("Renvoie `(rvec, tvec)`", t_form, True), ("Position retrouvée (2 ; −1 ; 40) cm", t_tvec, False),
                 ("La pose recolle aux coins (reprojection)", t_reproj, False), ("Utilise la taille L", t_L, False),
                 ("Accepte `dist` à zéro", t_dist, False), ("Accepte des coins de forme (1, 4, 2)", t_shape, False)])


SNIPPET_CHECKS = {
    "pixel": _snip_pixel, "rgb": _snip_rgb, "hsv": _snip_hsv, "split": _snip_split,
    "mask": _snip_mask, "objet": _snip_objet, "open": _snip_open, "close": _snip_close,
    "contours": _snip_contours, "objects": _snip_objects, "moments": _snip_moments, "centre": _snip_centre,
    "zpx": _snip_zpx, "objmask": _snip_objmask, "zobj": _snip_zobj, "xyz": _snip_xyz,
    "aruco": _snip_aruco, "obj3d": _snip_obj3d, "pnp": _snip_pnp,
}


_HSV_ORANGE = {"h_min": 5, "h_max": 20, "s_min": 120, "s_max": 255, "v_min": 80, "v_max": 255}


def _pf_scene(no_depth_obj=False, with_objects=True):
    """(tests) scène synthétique 240×320 : fond gris, trois objets orange (grand, moyen, petit) à des profondeurs différentes."""
    color = np.full((240, 320, 3), 90, np.uint8)
    depth = np.full((240, 320), 5000, np.uint16)
    if with_objects:
        orange = _bgr(12, 200, 220)
        for (v0, v1, u0, u1, z) in [(100, 160, 120, 200, 3000), (30, 80, 20, 70, 2000), (170, 200, 250, 280, 4000)]:
            color[v0:v1, u0:u1] = orange
            depth[v0:v1, u0:u1] = z
        if no_depth_obj:
            depth[30:80, 20:70] = 0            # le deuxième objet n'a aucune mesure
    return color, depth


def check_process_frame(fn):
    K1 = _K_SYN
    ds = 0.0001

    def run(color, depth, **kw):
        r = fn(color, depth, K1, ds, _HSV_ORANGE, **kw)
        _expect(r is not None, "la fonction renvoie `None` : pensez au `return`")
        _expect(isinstance(r, (list, tuple)), f"il faut renvoyer une **liste** de dictionnaires, pas un `{type(r).__name__}`")
        return list(r)

    def t_form():
        r = run(*_pf_scene())
        _expect(len(r) > 0, "aucun objet trouvé alors que la scène en contient trois")
        for o in r:
            _expect(isinstance(o, dict), "chaque objet doit être un **dictionnaire**")
            miss = {"contour", "centroid", "area", "depth_m", "xyz_m"} - set(o)
            _expect(not miss, f"clés manquantes : {sorted(miss)}")

    def t_count():
        r = run(*_pf_scene())
        _expect(len(r) == 3, f"3 objets attendus, {len(r)} trouvé(s) : traitez **tous** les contours, pas seulement le premier")

    def t_order():
        r = run(*_pf_scene())
        areas = [float(o["area"]) for o in r]
        _expect(len(r) == 3 and areas == sorted(areas, reverse=True), f"du plus grand au plus petit attendu, aires obtenues : {[round(a) for a in areas]}")

    def t_values():
        r = run(*_pf_scene())
        want = vision_ref.process_frame(*_pf_scene()[:1], _pf_scene()[1], K1, ds, _HSV_ORANGE)
        _expect(len(r) == len(want), f"{len(want)} objets attendus, {len(r)} obtenus")
        for i, (a, b) in enumerate(zip(r, want), 1):
            _expect(np.allclose(a["centroid"], b["centroid"], atol=1e-6), f"objet {i} : centroïde {np.round(a['centroid'], 2).tolist()} au lieu de {np.round(b['centroid'], 2).tolist()}")
            _expect(abs(a["depth_m"] - b["depth_m"]) < 1e-9, f"objet {i} : profondeur {a['depth_m']:.4f} m au lieu de {b['depth_m']:.4f} m (en mètres : `depth_scale` !)")
            _expect(np.allclose(np.asarray(a["xyz_m"], float), b["xyz_m"], atol=1e-6), f"objet {i} : position 3D {np.round(np.asarray(a['xyz_m'], float), 4).tolist()} au lieu de {np.round(b['xyz_m'], 4).tolist()}")
            _expect(abs(float(a["area"]) - b["area"]) < 1e-6, f"objet {i} : aire {a['area']} au lieu de {b['area']}")

    def t_nodepth():
        r = run(*_pf_scene(no_depth_obj=True))
        _expect(len(r) == 2, f"un objet sans aucune mesure de profondeur doit être **ignoré** : 2 objets attendus, {len(r)} obtenu(s)")

    def t_empty():
        try:
            r = run(*_pf_scene(with_objects=False))
        except AssertionError:
            raise
        except Exception as e:
            raise AssertionError(f"plante quand il n'y a aucun objet ({type(e).__name__} : {e}) : renvoyez une liste vide")
        _expect(len(r) == 0, f"aucun objet dans l'image : liste vide attendue, {len(r)} obtenu(s)")

    def t_minarea():
        c, d = _pf_scene()
        _expect(len(run(c, d, min_area=1000)) == 2, "`min_area` doit être transmis à `find_objects` (avec 1000 px², le plus petit objet disparaît)")

    def t_kernels():
        c, d = _pf_scene(with_objects=False)
        orange = _bgr(12, 200, 220)
        for u0, u1 in [(40, 100), (120, 180)]:           # deux objets séparés par un trou de 20 px
            c[100:160, u0:u1] = orange
            d[100:160, u0:u1] = 3000
        n_default = len(run(c, d))
        n_big = len(run(c, d, k_open=3, k_close=51))
        _expect(n_default == 2, f"réglages par défaut : 2 objets attendus, {n_default} obtenu(s)")
        _expect(n_big == 1, "`k_open` et `k_close` doivent être transmis à `clean_mask` : avec une fermeture de 51 px, les deux objets fusionnent en un seul")

    def t_input():
        c, d = _pf_scene()
        c0, d0 = c.copy(), d.copy()
        run(c, d)
        _expect(np.array_equal(c, c0) and np.array_equal(d, d0), "les images d'entrée ont été modifiées")

    return _run([("Renvoie une liste de dictionnaires complets", t_form, True), ("Trouve les 3 objets de la scène", t_count, False),
                 ("Du plus grand au plus petit", t_order, False), ("Centre, aire, profondeur et position 3D corrects", t_values, False),
                 ("Objet sans profondeur : ignoré", t_nodepth, False), ("Aucun objet : liste vide", t_empty, False),
                 ("`min_area` respecté", t_minarea, False), ("`k_open` / `k_close` transmis à `clean_mask`", t_kernels, False), ("Les images d'entrée ne sont pas modifiées", t_input, False)])


def check_snippet(sid, out, inputs):
    return SNIPPET_CHECKS[sid](out, inputs)


CHECKS = {
    "hsv_mask": check_hsv_mask,
    "clean_mask": check_clean_mask,
    "find_objects": check_find_objects,
    "centroid": check_centroid,
    "object_depth": check_object_depth,
    "deproject": check_deproject,
    "detect_aruco": check_detect_aruco,
    "marker_pose": check_marker_pose,
    "process_frame": check_process_frame,
}


# ---------------------------------------------------------------------------
# En terminal : python notebook/checks.py [module]
# ---------------------------------------------------------------------------

def main(module_name="ma_vision"):
    sys.path.insert(0, _ROOT)
    sys.path.insert(0, os.getcwd())
    try:
        mod = importlib.import_module(module_name)
    except ModuleNotFoundError:
        print(f"Module `{module_name}` introuvable : exportez-le depuis le notebook.")
        return 1
    bad = 0
    for name, check in CHECKS.items():
        fn = getattr(mod, name, None)
        if fn is None:
            print(f"\n{name}\n  ❓ absente de {module_name}.py")
            bad += 1
            continue
        res = check(fn)
        n_ok, n, good = summary(res)
        print(f"\n{name}  —  {n_ok}/{n}")
        for r in res:
            icon = {True: "✅", False: "❌", None: "⏭ "}[r.ok]
            print(f"  {icon} {r.label}" + (f"\n       → {r.detail}" if r.detail else ""))
        bad += 0 if good else 1
    print("\n" + ("Tout est vert." if not bad else f"{bad} fonction(s) à corriger."))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(*(sys.argv[1:2])))
