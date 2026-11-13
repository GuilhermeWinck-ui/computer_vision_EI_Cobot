import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full", app_title="TP1 — Vision par ordinateur")


@app.cell(hide_code=True)
def _():
    import json
    import math
    import os
    import sys
    import time

    import base64

    import cv2
    import marimo as _marimo
    import matplotlib.pyplot as plt
    import numpy as np

    try:
        import pyrealsense2 as rs
    except Exception:          # pas de SDK RealSense : seule la caméra est indisponible
        rs = None

    class _Mo:
        """Proxy de marimo : mo.image intègre l'image en base64 (évite les erreurs « Virtual file not found »)."""
        def __getattr__(self, name):
            return getattr(_marimo, name)

        @staticmethod
        def md(text, **_kw):
            # un md multi-paragraphes dans un hstack s'étalerait en colonnes : on l'enferme dans un vstack
            return _marimo.vstack([_marimo.md(text, **_kw)], gap=0)

        @staticmethod
        def image(src, width=None, **_kw):
            a = np.asarray(src)
            if a.dtype != np.uint8:
                a = np.clip(a, 0, 255).astype(np.uint8)
            if a.ndim == 3:
                ok, buf = cv2.imencode(".jpg", cv2.cvtColor(a, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92]); mime = "jpeg"
            else:
                ok, buf = cv2.imencode(".png", a); mime = "png"
            w = f' width="{int(width)}"' if width else ""
            return _marimo.Html(f'<img src="data:image/{mime};base64,{base64.b64encode(buf.tobytes()).decode()}"{w} style="max-width:100%;height:auto">')

    mo = _Mo()

    HERE = os.path.dirname(os.path.abspath(__file__))        # notebook/ : ce fichier, exercices.py, checks.py, mes_reponses.json
    ROOT = os.path.dirname(HERE)                             # dossier du TP : ma_vision.py, camera_utils.py, captures/, vision_ref
    CAPTURES = os.path.join(ROOT, "captures")
    os.makedirs(CAPTURES, exist_ok=True)
    DEFAULT_IMAGE = os.path.join(CAPTURES, "exemple.png")

    # modules du TP (même dossier que ce notebook) : énoncés, tests, solutions de référence
    for _p in (ROOT, HERE):
        if _p not in sys.path:
            sys.path.insert(0, _p)
    import checks
    import exercices
    import vision_ref

    return (
        CAPTURES,
        DEFAULT_IMAGE,
        HERE,
        ROOT,
        checks,
        cv2,
        exercices,
        json,
        math,
        mo,
        np,
        os,
        plt,
        rs,
        time,
        vision_ref,
    )


@app.cell(hide_code=True)
def _(CAPTURES, cv2, json, math, np, os, rs, time):
    # ------------------------------------------------------------------ images
    def to_rgb(bgr):
        """BGR (OpenCV) -> RGB (affichage). Accepte aussi une image 2D (masque)."""
        if bgr.ndim == 2:
            return cv2.cvtColor(bgr, cv2.COLOR_GRAY2RGB)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def as_bgr(img):
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img

    def estimate_K(w, h, hfov_deg=87.0):
        """Intrinsèques approchées quand elles ne sont pas fournies (champ horizontal 87° de la D405)."""
        f = (w / 2) / math.tan(math.radians(hfov_deg / 2))
        return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])

    def auto_pixel(bgr):
        """Pixel 'intéressant' par défaut : point le plus intérieur de la plus grande zone orange."""
        h, w = bgr.shape[:2]
        m = cv2.inRange(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV), (5, 100, 100), (20, 255, 255))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        if cv2.countNonZero(m) < 200:
            return w // 2, h // 2
        n, lab, st, _ = cv2.connectedComponentsWithStats(m)
        k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        d = cv2.distanceTransform((lab == k).astype(np.uint8), cv2.DIST_L2, 5)
        v, u = np.unravel_index(int(np.argmax(d)), d.shape)
        return int(u), int(v)

    # ------------------------------------------------------- chargement / save
    def load_frame(path):
        """Charge une image couleur + (si présents) sa profondeur et ses intrinsèques.

        Convention de fichiers (identique pour l'exemple et pour vos captures) :
            nom.png            image couleur BGR
            nom_depth.npy      profondeur brute uint16 (même taille que l'image)
            nom.json           intrinsèques (fx, fy, ppx, ppy, dist) + depth_scale
        """
        path = os.path.expanduser(str(path).strip())
        color = cv2.imread(path, cv2.IMREAD_COLOR)
        if color is None:
            raise ValueError(f"Impossible de lire l'image : {path}")
        h, w = color.shape[:2]
        stem = os.path.splitext(path)[0]
        depth, meta = None, {}
        if os.path.exists(stem + "_depth.npy"):
            d = np.load(stem + "_depth.npy")
            if d.shape == (h, w):
                depth = d.astype(np.uint16)
        if os.path.exists(stem + ".json"):
            with open(stem + ".json", encoding="utf-8") as fh:
                meta = json.load(fh)
        if "fx" in meta:
            K = np.array([[meta["fx"], 0, meta["ppx"]], [0, meta["fy"], meta["ppy"]], [0, 0, 1.0]])
            k_src = "fichier .json (capteur)"
        else:
            K = estimate_K(w, h)
            k_src = "estimées depuis le champ de vision 87° (pas de .json)"
        scale = meta.get("depth_scale")
        if depth is not None and scale is None:
            scale = 0.001
            k_src += " ; depth_scale absent : 0,001 supposé"
        return dict(color=color, depth=depth, K=K, dist=np.array(meta.get("dist", [0, 0, 0, 0, 0]), float),
                    depth_scale=scale, name=os.path.splitext(os.path.basename(path))[0],
                    origin=path, k_source=k_src)

    def save_frame(frame, name=""):
        """Enregistre une capture dans captures/ au format lu par load_frame."""
        os.makedirs(CAPTURES, exist_ok=True)
        stem = "".join(c for c in name.strip() if c.isalnum() or c in "-_") or time.strftime("capture_%Y%m%d_%H%M%S")
        if stem == "exemple":
            stem = "exemple_copie"
        base, i = os.path.join(CAPTURES, stem), 2
        while os.path.exists(base + ".png"):
            base = os.path.join(CAPTURES, f"{stem}_{i}")
            i += 1
        cv2.imwrite(base + ".png", frame["color"])
        if frame["depth"] is not None:
            np.save(base + "_depth.npy", frame["depth"])
        K = frame["K"]
        with open(base + ".json", "w", encoding="utf-8") as fh:
            json.dump(dict(width=int(frame["color"].shape[1]), height=int(frame["color"].shape[0]),
                           fx=float(K[0, 0]), fy=float(K[1, 1]), ppx=float(K[0, 2]), ppy=float(K[1, 2]),
                           dist=[float(x) for x in frame["dist"]], depth_scale=frame["depth_scale"],
                           source="capture RealSense D405"), fh, indent=2)
        return base + ".png"

    # ----------------------------------------------------------------- caméra
    def depth_to_color(depth, scale, near=0.05, far=0.60):
        """Profondeur -> image couleur : bleu = proche, rouge = loin, noir = invalide (0)."""
        d = depth.astype(np.float32) * float(scale)
        norm = np.clip((d - near) / (far - near), 0, 1)
        out = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        out[depth == 0] = 0
        return out

    def capture_from_camera():
        """Ouvre une fenêtre OpenCV avec le flux de la D405.
        ESPACE = capturer et fermer, ESC = annuler.
        Retourne (frame | None, message d'erreur | None)."""
        if rs is None:
            return None, "pyrealsense2 n'est pas installé : utilisez l'image d'exemple ou un fichier."
        win = "Capture  |  ESPACE = capturer    ESC = annuler"
        pipeline, cfg = rs.pipeline(), rs.config()
        cfg.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
        try:
            profile = pipeline.start(cfg)
        except Exception as exc:
            return None, f"Caméra introuvable ou occupée : {exc}"
        out, err = None, None
        try:
            scale = profile.get_device().first_depth_sensor().get_depth_scale()
            align = rs.align(rs.stream.color)
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win, 1300, 520)
            while True:
                aligned = align.process(pipeline.wait_for_frames())
                cf, df = aligned.get_color_frame(), aligned.get_depth_frame()
                if not cf or not df:
                    continue
                color, depth = np.asanyarray(cf.get_data()), np.asanyarray(df.get_data())
                disp = color.copy()
                cv2.putText(disp, "ESPACE : capturer    ESC : annuler", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 80), 2, cv2.LINE_AA)
                cv2.imshow(win, np.hstack([disp, depth_to_color(depth, scale)]))
                key = cv2.waitKey(1) & 0xFF
                if key == 32:
                    intr = cf.profile.as_video_stream_profile().get_intrinsics()
                    K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1.0]])
                    out = dict(color=color.copy(), depth=depth.copy(), K=K, dist=np.array(intr.coeffs, float),
                               depth_scale=float(scale), name="capture_camera", origin="caméra D405 (capture en direct)",
                               k_source="capteur (pyrealsense2)")
                    break
                if key == 27:
                    err = "Capture annulée."
                    break
        except Exception as exc:
            err = f"Erreur pendant la capture : {exc}"
        finally:
            try:
                pipeline.stop()
            except Exception:
                pass
            cv2.destroyAllWindows()
        return out, err

    # ---------------------------------------------------------------- dessins
    def hue_strip(intervals, width=900):
        """Bandeau des 180 teintes OpenCV avec les intervalles retenus entourés."""
        hues = np.arange(180, dtype=np.uint8).reshape(1, -1)
        strip = cv2.cvtColor(np.dstack([hues, np.full_like(hues, 255), np.full_like(hues, 255)]), cv2.COLOR_HSV2BGR)
        strip = cv2.resize(strip, (width, 44), interpolation=cv2.INTER_NEAREST)
        out = np.full((44 + 26, width, 3), 255, np.uint8)
        out[:44] = strip
        for lo_, hi_ in intervals:
            x0, x1 = int(lo_ / 180 * width), int((hi_ + 1) / 180 * width)
            cv2.rectangle(out, (x0, 1), (x1, 42), (255, 255, 255), 5)
            cv2.rectangle(out, (x0, 1), (x1, 42), (40, 40, 40), 2)
        for hh in range(0, 181, 30):
            x = min(int(hh / 180 * width), width - 1)
            cv2.line(out, (x, 44), (x, 50), (60, 60, 60), 1)
            cv2.putText(out, str(hh), (max(0, x - 10), 66), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1, cv2.LINE_AA)
        return out

    def zoom_patch(bgr, u, v, chan, n=7, cell=48):
        """Zoom n x n autour de (u, v) avec la valeur du canal choisi écrite dans chaque case."""
        r = n // 2
        pad = cv2.copyMakeBorder(bgr, r, r, r, r, cv2.BORDER_CONSTANT, value=(128, 128, 128))
        patch = pad[v:v + n, u:u + n]
        big = cv2.resize(patch, (n * cell, n * cell), interpolation=cv2.INTER_NEAREST)
        ci = {"B": 0, "G": 1, "R": 2}[chan]
        for i in range(n):
            for j in range(n):
                lum = float(patch[i, j].mean())
                cv2.putText(big, str(int(patch[i, j, ci])), (j * cell + 6, i * cell + cell // 2 + 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0) if lum > 140 else (255, 255, 255), 1, cv2.LINE_AA)
        for k in range(n + 1):
            cv2.line(big, (k * cell, 0), (k * cell, n * cell), (90, 90, 90), 1)
            cv2.line(big, (0, k * cell), (n * cell, k * cell), (90, 90, 90), 1)
        cv2.rectangle(big, (r * cell, r * cell), ((r + 1) * cell, (r + 1) * cell), (0, 0, 255), 3)
        return big

    def outlined_text(img, text, org, scale=0.6, color=(255, 255, 255), thick=2):
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (20, 20, 20), thick + 3, cv2.LINE_AA)
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

    def draw_markers(img, corners, ids, color=(0, 200, 0)):
        """Marqueurs ArUco bien visibles : bord épais à liseré blanc, coin n°1 repéré, id en gros."""
        if ids is None:
            return img
        for _c, _i in zip(corners, ids.ravel()):
            _p = _c.reshape(4, 2).astype(np.int32)
            cv2.polylines(img, [_p], True, (255, 255, 255), 9, cv2.LINE_AA)
            cv2.polylines(img, [_p], True, color, 4, cv2.LINE_AA)
            cv2.circle(img, tuple(int(t) for t in _p[0]), 9, (0, 0, 255), -1, cv2.LINE_AA)     # coin 1 (haut-gauche)
            _x, _y = int(_p[:, 0].min()), int(_p[:, 1].min())
            _txt = f"id {int(_i)}"
            (_tw, _th), _bl = cv2.getTextSize(_txt, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
            _x0, _y0 = max(0, _x), max(_th + 12, _y - 6)
            cv2.rectangle(img, (_x0, _y0 - _th - 8), (_x0 + _tw + 12, _y0 + 6), color, -1)          # pastille pleine : lisible sur tout fond
            cv2.putText(img, _txt, (_x0 + 6, _y0), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3, cv2.LINE_AA)
        return img

    def fr(x, n=2):
        """Nombre -> texte avec virgule décimale."""
        return f"{x:.{n}f}".replace(".", ",")

    return (
        auto_pixel,
        capture_from_camera,
        depth_to_color,
        draw_markers,
        fr,
        hue_strip,
        load_frame,
        save_frame,
        to_rgb,
        zoom_patch,
    )


@app.cell(hide_code=True)
def _(cv2, mo):
    # ------------------------------------------------------------ éléments d'interface
    # Blocs colorés : même couleur = même rôle, dans tout le notebook.
    #   rappel (bleu) · à vous (orange) · piège (rouge) · à retenir (vert)
    _KINDS = {
        "rappel": ("info", "📘 Rappel"),
        "vous": ("warn", "🎛️ À vous"),
        "coder": ("neutral", "✍️ À coder"),
        "piege": ("danger", "⚠️ Piège"),
        "retenir": ("success", "✅ À retenir"),
    }

    def block(kind, title, *body):
        """Bloc coloré avec titre ; body = textes markdown ou éléments marimo."""
        _k, _label = _KINDS[kind]
        _items = [mo.md(f"**{_label} — {title}**")] + [mo.md(b) if isinstance(b, str) else b for b in body]
        return mo.callout(mo.vstack(_items, gap=0.7), kind=_k)

    def figure(im_rgb, caption, width=330):
        """Image + légende, dans une colonne de largeur fixe (pas d'étalement)."""
        return mo.vstack([mo.image(im_rgb, width=width), mo.md(caption)], gap=0.3).style({"width": f"{width}px"})

    def png_uri(bgr):
        """Image BGR -> data URI PNG (sans perte : les valeurs affichées au survol sont exactes)."""
        import base64
        _ok, _buf = cv2.imencode(".png", bgr)
        return "data:image/png;base64," + base64.b64encode(_buf.tobytes()).decode()

    # Widget « cliquer sur l'image pour choisir un pixel » (nécessite : pip install anywidget)
    try:
        import anywidget
        import traitlets

        _ESM = """
        function render({ model, el }) {
          const w = model.get("w"), h = model.get("h"), width = model.get("width");
          const wrap = document.createElement("div");
          const canvas = document.createElement("canvas");
          canvas.width = w; canvas.height = h;
          canvas.style.cssText = `width:${width}px;height:${Math.round(width * h / w)}px;cursor:crosshair;border-radius:6px;border:1px solid #888;display:block;`;
          const ctx = canvas.getContext("2d");
          const raw = document.createElement("canvas"); raw.width = w; raw.height = h;
          const rctx = raw.getContext("2d", { willReadFrequently: true });
          const info = document.createElement("div");
          info.style.cssText = "font:12px monospace;margin-top:5px;min-height:1.5em;";
          wrap.appendChild(canvas); wrap.appendChild(info); el.appendChild(wrap);
          const img = new Image();
          const selected = () => `sélection : (u, v) = (${model.get("u")}, ${model.get("v")})`;
          const draw = () => {
            ctx.drawImage(raw, 0, 0);
            const u = model.get("u") + 0.5, v = model.get("v") + 0.5, k = w / width;
            ctx.lineWidth = 4 * k; ctx.strokeStyle = "#ffffff";
            ctx.beginPath(); ctx.arc(u, v, 14, 0, 2 * Math.PI); ctx.stroke();
            ctx.lineWidth = 2 * k; ctx.strokeStyle = "#ff2d2d";
            ctx.beginPath(); ctx.arc(u, v, 14, 0, 2 * Math.PI); ctx.stroke();
            info.textContent = selected();
          };
          const pos = (e) => {
            const r = canvas.getBoundingClientRect();
            return [Math.min(w - 1, Math.max(0, Math.floor((e.clientX - r.left) / r.width * w))),
                    Math.min(h - 1, Math.max(0, Math.floor((e.clientY - r.top) / r.height * h)))];
          };
          canvas.addEventListener("mousemove", (e) => {
            const [u, v] = pos(e); const d = rctx.getImageData(u, v, 1, 1).data;
            info.textContent = `survol : (u, v) = (${u}, ${v}) · RGB = (${d[0]}, ${d[1]}, ${d[2]})   — cliquez pour choisir`;
          });
          canvas.addEventListener("mouseleave", () => { info.textContent = selected(); });
          canvas.addEventListener("click", (e) => {
            const [u, v] = pos(e);
            model.set("u", u); model.set("v", v); model.save_changes(); draw();
          });
          img.onload = () => { rctx.drawImage(img, 0, 0); draw(); };
          img.src = model.get("image");
          model.on("change:u", draw); model.on("change:v", draw);
        }
        export default { render };
        """

        class PixelPicker(anywidget.AnyWidget):
            _esm = _ESM
            image = traitlets.Unicode("").tag(sync=True)
            w = traitlets.Int(1).tag(sync=True)
            h = traitlets.Int(1).tag(sync=True)
            width = traitlets.Int(560).tag(sync=True)
            u = traitlets.Int(0).tag(sync=True)
            v = traitlets.Int(0).tag(sync=True)

        _ESM_RECT = """
        function render({ model, el }) {
          const w = model.get("w"), h = model.get("h"), width = model.get("width");
          const wrap = document.createElement("div");
          const canvas = document.createElement("canvas");
          canvas.width = w; canvas.height = h;
          canvas.style.cssText = `width:${width}px;height:${Math.round(width * h / w)}px;cursor:crosshair;border-radius:6px;border:1px solid #888;display:block;user-select:none;`;
          const ctx = canvas.getContext("2d");
          const raw = document.createElement("canvas"); raw.width = w; raw.height = h;
          const rctx = raw.getContext("2d");
          const info = document.createElement("div");
          info.style.cssText = "font:12px monospace;margin-top:5px;min-height:1.5em;";
          wrap.appendChild(canvas); wrap.appendChild(info); el.appendChild(wrap);
          const img = new Image();
          const pos = (e) => {
            const r = canvas.getBoundingClientRect();
            return [Math.min(w - 1, Math.max(0, Math.round((e.clientX - r.left) / r.width * w))),
                    Math.min(h - 1, Math.max(0, Math.round((e.clientY - r.top) / r.height * h)))];
          };
          const draw = (x0, y0, x1, y1, tmp) => {
            ctx.drawImage(raw, 0, 0);
            const k = w / width;
            ctx.lineWidth = 4 * k; ctx.strokeStyle = "#ffffff"; ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
            ctx.lineWidth = 2 * k; ctx.strokeStyle = tmp ? "#ff9a2e" : "#111111"; ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
            info.textContent = `zone : u ∈ [${x0}, ${x1}], v ∈ [${y0}, ${y1}]  (${x1 - x0} × ${y1 - y0} px) — glissez pour tracer un rectangle sur l'objet`;
          };
          const cur = () => [model.get("u0"), model.get("v0"), model.get("u1"), model.get("v1")];
          let start = null;
          canvas.addEventListener("mousedown", (e) => { start = pos(e); e.preventDefault(); });
          window.addEventListener("mousemove", (e) => {
            if (!start) return;
            const p = pos(e);
            draw(Math.min(start[0], p[0]), Math.min(start[1], p[1]), Math.max(start[0], p[0]), Math.max(start[1], p[1]), true);
          });
          window.addEventListener("mouseup", (e) => {
            if (!start) return;
            const p = pos(e);
            const x0 = Math.min(start[0], p[0]), y0 = Math.min(start[1], p[1]);
            const x1 = Math.max(start[0], p[0]), y1 = Math.max(start[1], p[1]);
            start = null;
            if (x1 - x0 >= 4 && y1 - y0 >= 4) {
              model.set("u0", x0); model.set("v0", y0); model.set("u1", x1); model.set("v1", y1); model.save_changes();
            }
            draw(...cur(), false);
          });
          img.onload = () => { rctx.drawImage(img, 0, 0); draw(...cur(), false); };
          img.src = model.get("image");
          for (const k of ["u0", "v0", "u1", "v1"]) model.on("change:" + k, () => draw(...cur(), false));
        }
        export default { render };
        """

        class RectPicker(anywidget.AnyWidget):
            _esm = _ESM_RECT
            image = traitlets.Unicode("").tag(sync=True)
            w = traitlets.Int(1).tag(sync=True)
            h = traitlets.Int(1).tag(sync=True)
            width = traitlets.Int(440).tag(sync=True)
            u0 = traitlets.Int(0).tag(sync=True)
            v0 = traitlets.Int(0).tag(sync=True)
            u1 = traitlets.Int(10).tag(sync=True)
            v1 = traitlets.Int(10).tag(sync=True)
    except Exception:            # anywidget absent : on retombe sur des curseurs
        PixelPicker = RectPicker = None
    return PixelPicker, RectPicker, block, figure, png_uri


@app.cell(hide_code=True)
def _(HERE, ROOT, block, checks, cv2, exercices, json, mo, np, os, time, vision_ref):
    # ------------------------------------------------------------ mécanique des exercices de code
    import textwrap

    MA_VISION = os.path.join(ROOT, "ma_vision.py")

    # ---- sauvegarde automatique de votre travail (dans mes_reponses.json, à côté de vos réponses)
    #      image active, pixel, zone, réglages, et code de vos fonctions : tout est retrouvé au redémarrage
    def _etat_all():
        return _snip_saved().get("__etat__", {})

    def etat_get(key, default=None):
        return _etat_all().get(key, default)

    def etat_set(key, value):
        d = _snip_saved()
        e = d.setdefault("__etat__", {})
        if e.get(key) == value:
            return
        e[key] = value
        try:
            with open(SNIP_FILE, "w", encoding="utf-8") as f:
                json.dump(d, f, ensure_ascii=False, indent=1)
        except OSError:
            pass

    def ex_editor(name):
        """Éditeur de code : votre dernier brouillon (sauvegardé automatiquement), sinon votre dernière
        version exportée dans ma_vision.py, sinon l'énoncé."""
        _d = etat_get("fn:" + name)
        _code = None
        if isinstance(_d, dict) and _d.get("code"):
            try:
                _fresh = (not os.path.exists(MA_VISION)) or os.path.getmtime(MA_VISION) <= float(_d.get("t", 0))
            except OSError:
                _fresh = True
            if _fresh:                      # ma_vision.py modifié plus récemment (à la main) : il l'emporte
                _code = _d["code"]
        if _code is None:
            _code = exercices.load_code(name, MA_VISION)
        return mo.ui.code_editor(value=_code, language="python",
                                 min_height=100, debounce=True, show_copy_button=True)

    def ex_block(name, editor):
        """Énoncé : introduction, étapes, documentation, éditeur."""
        _ex = exercices.BY_NAME[name]
        _docs = " · ".join(f"[{exercices.DOC[k][0]}]({exercices.DOC[k][1]})" for k in _ex["docs"])
        _steps = "\n".join(f"{i}. {s}" for i, s in enumerate(_ex["steps"], 1))
        return block(
            "coder", f"{_ex['section']} · {_ex['title']}",
            _ex["intro"],
            "**Étapes**\n\n" + _steps,
            "📚 Documentation OpenCV : " + _docs,
            editor,
            "*Les tests se lancent quand vous cliquez **en dehors** de l'éditeur.*",
        )

    def _run_student(name, code, extra_ns=None):
        ns = {"cv2": cv2, "np": np, "__name__": "etudiant"}        # cv2 et np déjà importés pour vous
        ns.update(extra_ns or {})                                  # (section 8) vos fonctions déjà écrites
        try:
            exec(compile(code, f"<{name}>", "exec"), ns)
        except SyntaxError as e:
            return None, f"erreur de syntaxe, ligne {e.lineno} : {e.msg}"
        except Exception as e:
            return None, f"{type(e).__name__} : {e}"
        fn = ns.get(name)
        if not callable(fn):
            return None, f"la fonction `{name}` n'est pas définie : gardez la ligne `def {name}(...)`"
        return fn, None

    def evaluate(name, code, extra_ns=None):
        """Exécute le code de l'étudiant et lance les tests. `used` = sa fonction si elle est validée, sinon la référence."""
        _d = etat_get("fn:" + name)
        if not (isinstance(_d, dict) and _d.get("code") == code):
            etat_set("fn:" + name, {"code": code, "t": time.time()})
        fn, err = _run_student(name, code, extra_ns)
        res = checks.CHECKS[name](fn) if fn is not None else []
        good = fn is not None and checks.summary(res)[2]
        return {"name": name, "code": code, "fn": fn, "err": err, "results": res, "good": good,
                "used": fn if good else getattr(vision_ref, name)}

    def ex_view(ev, extra=None):
        """Résultat des tests, fonction utilisée pour la suite, solution à la demande."""
        name = ev["name"]
        if ev["fn"] is None:
            _status = mo.callout(mo.md("❌ **Le code ne s'exécute pas**\n\n```\n" + textwrap.fill(str(ev["err"])[:400], 88, replace_whitespace=False) + "\n```"), kind="danger")
            _table = mo.md("")
        else:
            _n_ok, _n, _good = checks.summary(ev["results"])
            _unwritten = ev["results"] and "pas encore écrite" in ev["results"][0].detail
            if _good:
                _status = mo.callout(mo.md(f"✅ **Validé : {_n_ok}/{_n} tests** — votre fonction est correcte."), kind="success")
            elif _unwritten:
                _status = mo.callout(mo.md("✍️ **Pas encore écrite** : remplacez `raise NotImplementedError` par votre code."), kind="neutral")
            else:
                _status = mo.callout(mo.md(f"🔧 **{_n_ok}/{_n} tests réussis** — ouvrez « Détail des tests » et lisez le premier ❌."), kind="warn")
            _icon = {True: "✅", False: "❌", None: "⏭️"}
            _rows = ["| | Test | Détail |", "|:-:|:--|:--|"]
            _rows += [f"| {_icon[r.ok]} | {r.label} | {r.detail} |" for r in ev["results"]]
            _table = mo.accordion({f"🧪 Détail des tests ({_n_ok}/{_n})": mo.md("\n".join(_rows))})
        _used = ("✅ **votre version**" if ev["good"] else "🛟 **la version de référence** (la vôtre n'est pas encore validée)")
        return mo.vstack([
            _status,
            _table,
            mo.md(extra) if extra else mo.md(""),
            mo.md(f"Utilisé pour la suite du notebook : {_used}"),
        ], gap=0.6)


    # ---- petits exercices : quelques lignes écrites sur les variables du notebook
    SNIP_FILE = os.path.join(HERE, "mes_reponses.json")

    def _snip_saved():
        try:
            with open(SNIP_FILE, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _snip_save(sid, code):
        d = _snip_saved()
        d[sid] = code
        try:
            with open(SNIP_FILE, "w", encoding="utf-8") as f:
                json.dump(d, f, ensure_ascii=False, indent=1)
        except OSError:
            pass

    def sn_editor(sid):
        """Éditeur d'un petit exercice (reprend votre dernière réponse enregistrée)."""
        _sp = exercices.SNIP_BY_ID[sid]
        return mo.ui.code_editor(value=_snip_saved().get(sid) or _sp["starter"], language="python",
                                 min_height=40, debounce=True, show_copy_button=False)

    def sn_block(sid, editor):
        _sp = exercices.SNIP_BY_ID[sid]
        _docs = " · ".join(f"[{exercices.DOC[k][0]}]({exercices.DOC[k][1]})" for k in _sp["docs"])
        _given = " · ".join(f"`{n}` ({d})" for n, d in _sp["given"])
        _out = " · ".join(f"`{n}` ({d})" for n, d in _sp["outputs"])
        return block(
            "coder", f"{_sp['section']} · {_sp['title']}",
            _sp["intro"],
            f"**Vous avez :** {_given} *(et `cv2`, `np`)*",
            f"**Vous écrivez :** {_out}",
            f"**Résultat attendu :** {_sp['expect']}",
            "📚 Documentation : " + _docs,
            editor,
            "*Les tests se lancent quand vous cliquez **en dehors** de l'éditeur.*",
        )

    def sn_eval(sid, code, **inputs):
        """Exécute votre code sur des COPIES des variables du notebook, puis lance les tests."""
        _sp = exercices.SNIP_BY_ID[sid]
        _ns = {"cv2": cv2, "np": np, "__name__": "etudiant"}
        for _k, _v in inputs.items():
            _ns[_k] = _v.copy() if isinstance(_v, np.ndarray) else _v
        _err = _hint = None
        try:
            exec(compile(code, f"<{sid}>", "exec"), _ns)
        except SyntaxError as e:
            _err = f"erreur de syntaxe, ligne {e.lineno} : {e.msg}"
        except Exception as e:
            _err = f"{type(e).__name__} : {getattr(e, 'msg', None) or e}"
            if isinstance(e, (TypeError, cv2.error)):
                _hint = "Relisez la signature de la fonction dans la documentation : arguments obligatoires et noms des paramètres (par exemple `mask=`)."
        _ref = vision_ref.SNIPPET_REF[sid](**inputs)
        _names = [n.strip() for o, _ in _sp["outputs"] for n in o.split(",")]
        _out = {n: _ns.get(n) for n in _names}
        # ---- le même code est aussi essayé sur quelques cas synthétiques à réponse connue
        _alts = []
        if _err is None:
            for _label, _ai in checks.alt_inputs(sid, inputs):
                _n2 = {"cv2": cv2, "np": np, "__name__": "etudiant"}
                for _k, _v in _ai.items():
                    _n2[_k] = _v.copy() if isinstance(_v, np.ndarray) else _v
                _e2 = None
                try:
                    exec(compile(code, f"<{sid}>", "exec"), _n2)
                except Exception as e:
                    _e2 = f"{type(e).__name__} : {getattr(e, 'msg', None) or e}"
                _alts.append({"label": _label, "inputs": _ai, "err": _e2, "out": {n: _n2.get(n) for n in _names}})
        _res = [] if _err else checks.check_snippet(sid, _out, {**inputs, "_alts": _alts})
        _good = _err is None and checks.summary(_res)[2]
        _unwritten = code.strip() == _sp["starter"].strip()
        if not _unwritten:
            _snip_save(sid, code)
        return {"sid": sid, "err": _err, "hint": _hint, "results": _res, "good": _good, "unwritten": _unwritten,
                "out": _out, "values": _out if _good else _ref}

    def sn_view(ev, extra=None):
        """Verdict, tableau de tests, et ce que la suite du notebook utilise."""
        if ev["err"]:
            _status = mo.callout(mo.md("❌ **Le code ne s'exécute pas**\n\n```\n" + textwrap.fill(str(ev["err"])[:400], 88, replace_whitespace=False) + "\n```"
                                       + (f"\n\n💡 {ev['hint']}" if ev.get("hint") else "")), kind="danger")
            _table = mo.md("")
        elif ev["unwritten"]:
            _status = mo.callout(mo.md("✍️ **Pas encore écrit** : remplacez `None` par votre code."), kind="neutral")
            _table = mo.md("")
        else:
            _n_ok, _n, _good = checks.summary(ev["results"])
            if _good:
                _status = mo.callout(mo.md(f"✅ **Validé : {_n_ok}/{_n} tests.**"), kind="success")
            else:
                _status = mo.callout(mo.md(f"🔧 **{_n_ok}/{_n} tests réussis** — ouvrez « Détail des tests » et lisez le premier ❌."), kind="warn")
            _icon = {True: "✅", False: "❌", None: "⏭️"}
            _rows = ["| | Test | Détail |", "|:-:|:--|:--|"]
            _rows += [f"| {_icon[r.ok]} | {r.label} | {r.detail} |" for r in ev["results"]]
            _table = mo.accordion({f"🧪 Détail des tests ({_n_ok}/{_n})": mo.md("\n".join(_rows))})
        _used = "✅ **votre résultat**" if ev["good"] else "🛟 **le résultat attendu** (le vôtre n'est pas encore validé)"
        _ex = mo.md(extra) if isinstance(extra, str) else (extra if extra is not None else mo.md(""))
        return mo.vstack([_status, _table, _ex, mo.md(f"La suite du notebook utilise : {_used}")], gap=0.6)

    return (
        MA_VISION,
        etat_get,
        etat_set,
        evaluate,
        ex_block,
        ex_editor,
        ex_view,
        sn_block,
        sn_editor,
        sn_eval,
        sn_view,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 🤖 TP1 — Vision par ordinateur : voir un objet coloré
    *EI Cobot · École Centrale de Lille · RealSense D405 · Python · OpenCV*

    Ce notebook suit **le même fil que les slides** : on part d'une image et on arrive à
    la position 3D d'un objet et d'un marqueur ArUco.

    | # | Section | Question | Slides |
    |---|---------|----------|--------|
    | 1 | Image | Qu'est-ce qu'un pixel pour l'ordinateur ? | 3 – 8 |
    | 2 | HSV | Comment séparer la couleur de l'éclairage ? | 9 – 13 |
    | 3 | Seuillage et masque | Quels pixels appartiennent à l'objet ? | 14 – 19 |
    | 4 | Contour | Quelle est la frontière de l'objet ? | 20 – 24 |
    | 5 | Centroïde | Où est « le » point de l'objet ? | 25 – 28 |
    | 6 | Profondeur | À quelle distance ? Et donc où dans l'espace ? | 29 – 34 |
    | 7 | ArUco | Un repère de position connue dans la scène | 35 – 39 |

    > 💡 **Comment travailler** : les blocs **🎛️ À vous** se manipulent avec des curseurs ; les blocs **✍️ À coder** sont à **écrire vous-même** :
    > tapez votre code, cliquez en dehors de l'éditeur, les tests se lancent. Ce que vous écrivez alimente la suite du notebook,
    > puis devient votre module `ma_vision.py` (celui de `vision_node.py`).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # 📥 Source de l'image
    L'image choisie ici est utilisée par **toutes** les sections. Par défaut, vous travaillez sur
    l'**image d'exemple** fournie : le résultat attendu est connu (voir la section Synthèse).
    Pour travailler sur **votre** image : cochez-la dans l'explorateur, collez son chemin, ou prenez une photo avec la caméra.
    """)
    return


@app.cell(hide_code=True)
def _(DEFAULT_IMAGE, etat_get, mo, os):
    _saved = etat_get("image")
    _start = _saved if (isinstance(_saved, str) and os.path.exists(_saved)) else DEFAULT_IMAGE
    get_path, set_path = mo.state(_start)          # LE fichier image actif (dernière utilisée, sinon l'exemple)
    get_msg, set_msg = mo.state("")                # message d'état (capture caméra)
    return get_msg, get_path, set_msg, set_path


@app.cell(hide_code=True)
def _(CAPTURES, mo, set_path):
    # L'explorateur ne fait que REMPLIR le champ « chemin » ci-dessous : une seule source, pas de mode à choisir.
    def _on_pick(sel):
        if sel:
            set_path(sel[0].path)

    browser = mo.ui.file_browser(
        initial_path=CAPTURES, filetypes=[".png", ".jpg", ".jpeg", ".bmp"],
        selection_mode="file", multiple=False, restrict_navigation=False,
        label="Parcourir les fichiers (cochez une image)", on_change=_on_pick,
    )
    return (browser,)


@app.cell(hide_code=True)
def _(DEFAULT_IMAGE, mo, set_path):
    reset_btn = mo.ui.button(
        label="↩ Revenir à l'image d'exemple",
        on_click=lambda _: set_path(DEFAULT_IMAGE),
    )
    return (reset_btn,)


@app.cell(hide_code=True)
def _(browser, get_path, mo, reset_btn, set_path):
    path_text = mo.ui.text(value=get_path(), label="**Image utilisée** (chemin)", full_width=True,
                           on_change=set_path)
    mo.vstack([
        path_text,
        reset_btn,
        mo.md("*Collez un chemin (puis Entrée) ou cochez une image dans l'explorateur : le champ se met à jour. "
              "Si `nom_depth.npy` et `nom.json` existent à côté de `nom.png`, la profondeur et les "
              "intrinsèques sont chargées avec.*"),
        mo.accordion({"📂 Parcourir les fichiers": browser}),
    ])
    return (path_text,)


@app.cell(hide_code=True)
def _(mo):
    save_name = mo.ui.text(value="", placeholder="optionnel, ex. mon_objet_01", label="Nom de la capture")
    save_name
    return (save_name,)


@app.cell(hide_code=True)
def _(capture_from_camera, mo, save_frame, save_name, set_msg, set_path):
    def _on_capture(_):
        _frame, _err = capture_from_camera()
        if _frame is None:
            set_msg("⚠️ " + (_err or "Aucune image capturée."))
        else:
            _p = save_frame(_frame, save_name.value)          # enregistrée automatiquement dans captures/
            set_path(_p)                                      # et devient l'image utilisée
            set_msg(f"✅ Capture enregistrée et chargée : `{_p}` (+ `_depth.npy` et `.json`).")

    cam_btn = mo.ui.button(label="📷 Prendre une photo avec la D405", kind="success", on_click=_on_capture)
    mo.hstack([
        cam_btn,
        mo.md("Une fenêtre OpenCV s'ouvre : **ESPACE** = capturer, **ESC** = annuler. "
              "La photo est **enregistrée dans `captures/`** et devient l'image utilisée."),
    ], justify="start", align="center")
    return


@app.cell(hide_code=True)
def _(get_msg, mo):
    mo.callout(mo.md(get_msg()), kind="neutral") if get_msg() else None
    return


@app.cell(hide_code=True)
def _(DEFAULT_IMAGE, etat_set, load_frame, mo, path_text):
    _problem = None
    frame = None
    try:
        frame = load_frame(path_text.value)
    except Exception as _exc:
        _problem = str(_exc)
    if frame is None:
        frame = load_frame(DEFAULT_IMAGE)
        _problem = (_problem or "Source indisponible") + " → image d'exemple utilisée."
    source_problem = _problem
    if not _problem:
        etat_set("image", frame["origin"])
    mo.callout(mo.md("⚠️ " + source_problem), kind="warn") if source_problem else None
    return (frame,)


@app.cell(hide_code=True)
def _(fr, frame, mo, to_rgb):
    _c, _d, _K = frame["color"], frame["depth"], frame["K"]
    _depth_txt = "❌ aucune" if _d is None else (f"✅ `{_d.shape}` {_d.dtype}, `depth_scale = {frame['depth_scale']}` m / unité")
    mo.hstack([
        mo.image(to_rgb(_c), width=520),
        mo.md(f"""
        **Image active : `{frame['name']}`**

        | | |
        |---|---|
        | Origine | `{frame['origin']}` |
        | Image couleur | `{_c.shape}` {_c.dtype} (BGR) |
        | Profondeur | {_depth_txt} |
        | Intrinsèques | f<sub>x</sub> = {fr(_K[0,0],1)} · f<sub>y</sub> = {fr(_K[1,1],1)} · c<sub>x</sub> = {fr(_K[0,2],1)} · c<sub>y</sub> = {fr(_K[1,2],1)} px |
        | Provenance de K | {frame['k_source']} |
        """),
    ], justify="start", gap=2)
    return


@app.cell(hide_code=True)
def _(block, mo):
    mo.vstack([
        mo.md("---\n# 1 · Image numérique"),
        block("rappel", "une image est un tableau NumPy",
              """
    - `img[v, u]` donne **3 entiers** entre 0 et 255, dans l'ordre **B, G, R** (et non R, G, B).
    - OpenCV et les dessins raisonnent en **(u, v)** : *u* = colonne →, *v* = ligne ↓.
    - NumPy indexe en **[v, u]** : la **ligne d'abord**.
                  """),
    ])
    return


@app.cell(hide_code=True)
def _(frame):
    img = frame["color"]
    H_img, W_img = img.shape[:2]
    return H_img, W_img, img


@app.cell(hide_code=True)
def _(H_img, PixelPicker, W_img, auto_pixel, etat_get, frame, img, mo, png_uri):
    _u0, _v0 = auto_pixel(img)
    _s = etat_get("pixel")
    if isinstance(_s, dict) and _s.get("image") == frame["origin"]:        # pixel déjà choisi sur cette image
        _u0, _v0 = min(max(int(_s["uv"][0]), 0), W_img - 1), min(max(int(_s["uv"][1]), 0), H_img - 1)
    if PixelPicker is not None:
        picker = mo.ui.anywidget(PixelPicker(image=png_uri(img), w=W_img, h=H_img, width=560, u=int(_u0), v=int(_v0)))
        px_u = px_v = None
    else:      # anywidget non installé : curseurs
        picker = None
        px_u = mo.ui.slider(0, W_img - 1, value=_u0, label="u (colonne →)", show_value=True)
        px_v = mo.ui.slider(0, H_img - 1, value=_v0, label="v (ligne ↓)", show_value=True)
    px_chan = mo.ui.dropdown(["B", "G", "R"], value="R", label="Canal écrit dans la grille")
    return picker, px_chan, px_u, px_v


@app.cell(hide_code=True)
def _(H_img, W_img, etat_set, frame, picker, px_u, px_v):
    # (u, v) : le pixel choisi, réutilisé par toutes les sections suivantes
    if picker is not None:
        pu, pv = int(picker.value["u"]), int(picker.value["v"])
    else:
        pu, pv = int(px_u.value), int(px_v.value)
    pu, pv = min(max(pu, 0), W_img - 1), min(max(pv, 0), H_img - 1)
    etat_set("pixel", {"image": frame["origin"], "uv": [pu, pv]})
    return pu, pv


@app.cell(hide_code=True)
def _(
    block,
    img,
    mo,
    np,
    picker,
    pu,
    pv,
    px_chan,
    px_u,
    px_v,
    to_rgb,
    zoom_patch,
):
    # ---- lecture d'un pixel : ATTENTION à l'ordre [v, u] puis (B, G, R)
    _b, _g, _r = (int(x) for x in img[pv, pu])
    _swatch = np.full((70, 70, 3), (_b, _g, _r), np.uint8)
    _left = picker if picker is not None else mo.vstack([
        mo.callout(mo.md("`pip install anywidget` pour pouvoir **cliquer sur l'image** ; en attendant, utilisez les curseurs."), kind="warn"),
        px_u, px_v,
        mo.image(to_rgb(img), width=560),
    ])
    block(
        "vous", "cliquez sur l'image pour choisir un pixel",
        "*Il sera réutilisé dans **toutes** les sections suivantes. Survolez l'image : les valeurs RGB s'affichent.*",
        mo.hstack([
            _left,
            mo.vstack([
                mo.image(to_rgb(zoom_patch(img, pu, pv, px_chan.value)), width=336),
                px_chan,
            ], gap=0.4),
            mo.vstack([
                mo.image(to_rgb(_swatch), width=70),
                mo.md(f"Pixel choisi\n\n**(u, v) = ({pu}, {pv})**"),
            ], gap=0.4),
        ], justify="start", gap=1.5, align="start", wrap=True),
    )
    return


@app.cell(hide_code=True)
def _(sn_block, sn_editor):
    sed_pixel = sn_editor("pixel")
    sn_block("pixel", sed_pixel)
    return (sed_pixel,)


@app.cell(hide_code=True)
def _(img, pu, pv, sed_pixel, sn_eval, sn_view):
    _ev = sn_eval("pixel", sed_pixel.value, img=img, pu=pu, pv=pv)
    _o = _ev["out"]
    _extra = None
    if _ev["good"]:
        _extra = (f"Votre code a lu **B = {int(_o['b'])}, G = {int(_o['g'])}, R = {int(_o['r'])}** au pixel ({pu}, {pv}) : "
                  f"en RGB c'est **({int(_o['r'])}, {int(_o['g'])}, {int(_o['b'])})**, la couleur que l'image affiche.")
    sn_view(_ev, _extra)
    return


@app.cell(hide_code=True)
def _(sn_block, sn_editor):
    sed_rgb = sn_editor("rgb")
    sn_block("rgb", sed_rgb)
    return (sed_rgb,)


@app.cell(hide_code=True)
def _(figure, img, mo, sed_rgb, sn_eval, sn_view):
    _ev = sn_eval("rgb", sed_rgb.value, img=img)
    _extra = None
    if _ev["good"]:
        _extra = mo.hstack([
            figure(img, "*`img` affichée telle quelle : canaux B et R inversés (faux)*", 360),
            figure(_ev["values"]["rgb"], "*Votre `rgb` (correct)*", 360),
        ], justify="start", gap=1.2, wrap=True)
    sn_view(_ev, _extra)
    return


@app.cell(hide_code=True)
def _(mo):
    delta = mo.ui.slider(0, 255, value=100, label="Valeur ajoutée à chaque canal", show_value=True)
    gain = mo.ui.slider(0.3, 1.8, step=0.05, value=0.5, label="Éclairage (gain d'exposition)", show_value=True)
    return delta, gain


@app.cell(hide_code=True)
def _(block, cv2, delta, figure, fr, gain, img, mo, np, pu, pv, to_rgb):
    # ---- piège 2 : uint8
    _d = int(delta.value)
    _wrap = img + np.uint8(_d)                              # débordement (modulo 256)
    _sat = cv2.add(img, np.full_like(img, _d))              # saturation à 255
    _rr = int(img[pv, pu, 2])
    # ---- éclairage
    _dark = cv2.convertScaleAbs(img, alpha=float(gain.value))
    _b0, _g0, _r0 = (int(x) for x in img[pv, pu])
    _b1, _g1, _r1 = (int(x) for x in _dark[pv, pu])

    mo.ui.tabs({
        "⚠️ Piège 2 · uint8": block(
            "piege", "le type uint8",
            "Un pixel est un entier de **0 à 255**. Au-delà, NumPy **revient à 0** (modulo 256) ; `cv2.add` **sature** à 255.",
            delta,
            mo.hstack([
                figure(to_rgb(img), "*Original*"),
                figure(to_rgb(_wrap), "*`img + d` : modulo 256*"),
                figure(to_rgb(_sat), "*`cv2.add(img, d)` : saturé*"),
            ], justify="start", gap=1.2, wrap=True),
            f"Au pixel ({pu}, {pv}), canal R : **{_rr} + {_d}** → NumPy : **{(_rr + _d) % 256}** · cv2.add : **{min(_rr + _d, 255)}**",
        ),
        "💡 Éclairage": block(
            "vous", "un pixel n'est pas « la couleur de l'objet »",
            "Même scène, même point : quand l'éclairage change, **toutes** les valeurs changent. "
            "Un seuil sur R, G ou B est donc fragile → il faut un espace où l'éclairage est séparé de la couleur.",
            gain,
            mo.hstack([
                figure(to_rgb(img), "*Normal*", 420),
                figure(to_rgb(_dark), f"*Gain × {fr(gain.value)}*", 420),
            ], justify="start", gap=1.5, wrap=True),
            f"""
    | Pixel ({pu}, {pv}) | Normal | Gain × {fr(gain.value)} | Change ? |
    |---|---|---|---|
    | **R** | {_r0} | {_r1} | oui |
    | **G** | {_g0} | {_g1} | oui |
    | **B** | {_b0} | {_b1} | oui |

    👉 Trois nombres changent à la fois : on verra en **section 2** un espace où la lumière n'affecte qu'un seul nombre.
                """,
        ),
    })
    return


@app.cell(hide_code=True)
def _(block):
    block(
        "retenir", "section 1",
        """
    - `img[v, u]` : **ligne d'abord**, puis 3 valeurs **B, G, R**.
    - Les afficheurs attendent du **RGB** : on convertit avant d'afficher.
    - `uint8` : `+` boucle à 0, `cv2.add` sature à 255.
    - L'éclairage change les 3 canaux → on passera en **HSV** (section 2).
            """,
    )
    return


@app.cell(hide_code=True)
def _(block, mo):
    mo.vstack([
        mo.md("---\n# 2 · L'espace HSV"),
        block("rappel", "séparer la couleur de l'éclairage",
              """
    Un cône : **H** (teinte) = angle, **S** (saturation) = rayon, **V** (valeur = luminosité) = hauteur.
    En OpenCV (8 bits) : **H ∈ [0, 179]**, **S, V ∈ [0, 255]**.
                  """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    gain2 = mo.ui.slider(0.3, 1.8, step=0.05, value=0.5, label="Éclairage (gain d'exposition)", show_value=True)
    return (gain2,)


@app.cell(hide_code=True)
def _(block, cv2, figure, fr, gain2, img, mo, pu, pv, to_rgb):
    _dark = cv2.convertScaleAbs(img, alpha=float(gain2.value))
    _a = [int(x) for x in img[pv, pu]][::-1]                                     # R, G, B
    _b = [int(x) for x in _dark[pv, pu]][::-1]
    _ha = [int(x) for x in cv2.cvtColor(img[pv:pv + 1, pu:pu + 1], cv2.COLOR_BGR2HSV)[0, 0]]
    _hb = [int(x) for x in cv2.cvtColor(_dark[pv:pv + 1, pu:pu + 1], cv2.COLOR_BGR2HSV)[0, 0]]
    _row = lambda name, k, va, vb: f"| **{name}** | {va[k]} | {vb[k]} | {vb[k] - va[k]:+d} |"
    block(
        "vous", "la même expérience qu'en section 1, vue en HSV",
        "Changez l'éclairage : **R, G et B bougent tous les trois**. En HSV, seul **V** suit la lumière ; **H** reste (presque) fixe.",
        gain2,
        mo.hstack([
            figure(to_rgb(img), "*Normal*", 380),
            figure(to_rgb(_dark), f"*Gain × {fr(gain2.value)}*", 380),
            mo.md(f"""
    | Pixel ({pu}, {pv}) | Normal | Gain × {fr(gain2.value)} | Écart |
    |---|---|---|---|
    {_row('R', 0, _a, _b)}
    {_row('G', 1, _a, _b)}
    {_row('B', 2, _a, _b)}
    {_row('H', 0, _ha, _hb)}
    {_row('S', 1, _ha, _hb)}
    {_row('V', 2, _ha, _hb)}
                """),
        ], justify="start", gap=1.5, wrap=True, align="start"),
        "👉 C'est pourquoi on seuille sur **H** (et S, V pour écarter le gris et le noir) : un seuil sur H résiste aux changements de lumière.",
    )
    return


@app.cell(hide_code=True)
def _(sn_block, sn_editor):
    sed_hsv = sn_editor("hsv")
    sn_block("hsv", sed_hsv)
    return (sed_hsv,)


@app.cell(hide_code=True)
def _(img, sed_hsv, sn_eval, sn_view):
    sev_hsv = sn_eval("hsv", sed_hsv.value, img=img)
    hsv = sev_hsv["values"]["hsv"]
    hsv_ok = sev_hsv["good"]
    sn_view(sev_hsv)
    return hsv, hsv_ok


@app.cell(hide_code=True)
def _(sn_block, sn_editor):
    sed_split = sn_editor("split")
    sn_block("split", sed_split)
    return (sed_split,)


@app.cell(hide_code=True)
def _(hsv, sed_split, sn_eval, sn_view):
    sev_split = sn_eval("split", sed_split.value, hsv=hsv)
    H_ch, S_ch, V_ch = (sev_split["values"][k] for k in "HSV")
    split_ok = sev_split["good"]
    sn_view(sev_split)
    return H_ch, S_ch, V_ch, split_ok


@app.cell(hide_code=True)
def _(
    H_ch,
    S_ch,
    V_ch,
    block,
    cv2,
    figure,
    hsv,
    hsv_ok,
    img,
    mo,
    np,
    pu,
    pv,
    split_ok,
    to_rgb,
):
    mo.stop(not (hsv_ok and split_ok), mo.callout(mo.md("🔒 *Les résultats s'afficheront dès que **2.1** et **2.2** seront validés.*"), kind="neutral"))
    _h, _s, _v = H_ch, S_ch, V_ch                      # vos trois canaux
    _h_col = cv2.cvtColor(cv2.merge([_h, np.full_like(_h, 255), np.full_like(_h, 255)]), cv2.COLOR_HSV2BGR)
    _hh, _ss, _vv = (int(x) for x in hsv[pv, pu])
    _bb, _gg, _rr = (int(x) for x in img[pv, pu])
    block(
        "vous", "les trois canaux H, S, V de votre image",
        mo.hstack([
            figure(to_rgb(img), "*Image*", 300),
            figure(to_rgb(_h_col), "**H**: Teinte : la couleur « pure », indépendante de la luminosité", 300),
            figure(_s, "**S** : Saturation : élevée = couleur vive, faible = gris ou blanc", 300),
            figure(_v, "**V** : Valeur : la luminosité (ombres et reflets)", 300),
        ], justify="start", gap=1, wrap=True),
        f"Pixel choisi ({pu}, {pv}) : **BGR = ({_bb}, {_gg}, {_rr})** → **HSV = ({_hh}, {_ss}, {_vv})**",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    s_low = mo.ui.slider(0, 120, value=40, label="« S faible » si S <", show_value=True)
    return (s_low,)


@app.cell(hide_code=True)
def _(S_ch, V_ch, block, hue_strip, img, mo, np, s_low, to_rgb):
    _S, _V = S_ch, V_ch
    _weak = _S < s_low.value
    _spec = (_V > 235) & (_S < 60)
    _view = img.copy()
    _view[_weak] = (0.45 * _view[_weak] + 0.55 * np.array([255, 0, 255])).astype(np.uint8)    # magenta
    _view[_spec] = (0.3 * _view[_spec] + 0.7 * np.array([255, 255, 0])).astype(np.uint8)      # cyan
    mo.ui.tabs({
        "⚠️ Piège 1 · le rouge est coupé en deux": block(
            "piege", "le rouge est aux deux extrémités des teintes",
            "H ≈ 0 et H ≈ 179 sont la **même couleur** (le cercle des teintes se referme). "
            "Pour détecter un objet rouge il faut **deux intervalles** de H, combinés par un OU.",
            mo.image(to_rgb(hue_strip([(0, 10), (170, 179)])), width=800),
            "*Les 180 teintes d'OpenCV, avec les deux bouts du rouge entourés.*",
        ),
        "⚠️ Piège 2 · S faible, H n'a plus de sens": block(
            "piege", "gris, blanc et reflets : la teinte n'est pas fiable",
            "Quand la saturation est faible, la teinte est quasi aléatoire. Les **reflets** (V fort, S faible) perdent aussi leur couleur : ils feront des **trous dans le masque**.",
            s_low,
            mo.hstack([
                mo.image(to_rgb(_view), width=520),
                mo.md(f"""
    🟣 **S < {s_low.value}** : teinte non fiable — **{100 * _weak.mean():.1f} %** des pixels

    🔵 **Reflets** (V > 235 et S < 60) : **{100 * _spec.mean():.1f} %** des pixels
                    """),
            ], justify="start", gap=1.5, align="center", wrap=True),
        ),
    })
    return


@app.cell(hide_code=True)
def _(fr, hsv, img, mo, pu, pv):
    # ---- la conversion à la main (formules de la slide 12) pour le pixel choisi
    _b, _g, _r = (float(x) for x in img[pv, pu])
    _V = max(_r, _g, _b)
    _D = _V - min(_r, _g, _b)
    _S = 0.0 if _V == 0 else _D / _V
    if _D == 0:
        _Hdeg = 0.0
    elif _V == _r:
        _Hdeg = 60 * (_g - _b) / _D
    elif _V == _g:
        _Hdeg = 60 * (2 + (_b - _r) / _D)
    else:
        _Hdeg = 60 * (4 + (_r - _g) / _D)
    _Hdeg %= 360
    _hand = (round(_Hdeg / 2) % 180, round(_S * 255), int(_V))
    _cv = tuple(int(x) for x in hsv[pv, pu])
    _ok = all(abs(a - b) <= 1 for a, b in zip(_hand, _cv))
    mo.accordion({
        "🔍 Sous le capot : la conversion BGR → HSV à la main (slide 12)": mo.md(f"""
    (R, G, B) = ({int(_r)}, {int(_g)}, {int(_b)}) au pixel ({pu}, {pv})

    | | calcul | à la main | `cv2.cvtColor` |
    |---|---|---|---|
    | **V** | max(R, G, B) | {int(_V)} | {_cv[2]} |
    | **S** | Δ / V × 255, Δ = max − min = {int(_D)} | {_hand[1]} | {_cv[1]} |
    | **H** | {fr(_Hdeg, 1)}° ÷ 2 (OpenCV code H sur 0-179) | {_hand[0]} | {_cv[0]} |

    {"✅ identique (à ±1 près : arrondis)" if _ok else "⚠️ écart > 1 : vérifiez l'ordre des canaux"}
            """),
    })
    return


@app.cell(hide_code=True)
def _(block):
    block(
        "retenir", "section 2",
        """
    - En OpenCV (8 bits) : **H ∈ [0, 179]**, pas 360 ; S et V ∈ [0, 255].
    - Un changement d'éclairage touche surtout **V** : on seuille sur **H** (plus S et V pour écarter gris et noir).
    - Rouge : **deux** intervalles de H. Reflets et gris : S faible, teinte non fiable.
            """,
    )
    return


@app.cell(hide_code=True)
def _(block, mo):
    mo.vstack([
        mo.md("---\n# 3 · Seuillage et masque"),
        block("rappel", "un test sur chaque pixel",
              r"""
    $$M(u,v) = 255 \ \text{si}\ \text{bas} \le H(u,v) \le \text{haut},\quad 0 \ \text{sinon}$$

    (et de même sur S et V : les 3 tests doivent être vrais). Le résultat, le **masque**, est un tableau de **0 ou 255** (`uint8`) de la même taille que l'image.
    Trois étapes : **3.1** trouver les bornes · **3.2** appliquer · **3.3** nettoyer.
                  """),
    ])
    return


@app.cell(hide_code=True)
def _(etat_get, mo):
    # bornes courantes (H min, H max, S min, V min) : le bouton « Appliquer » les modifie
    # (au démarrage : celles de votre dernière session)
    get_bounds, set_bounds = mo.state(tuple(int(etat_get(k, d)) for k, d in
                                            (("r:h_min", 5), ("r:h_max", 20), ("r:s_min", 100), ("r:v_min", 100))))
    return get_bounds, set_bounds


@app.cell(hide_code=True)
def _(H_img, RectPicker, W_img, auto_pixel, etat_get, frame, img, mo, png_uri):
    _u0, _v0 = auto_pixel(img)
    _x0, _y0 = max(0, _u0 - 40), max(0, _v0 - 30)
    _x1, _y1 = min(W_img - 1, _x0 + 80), min(H_img - 1, _y0 + 60)
    _s = etat_get("roi")
    if isinstance(_s, dict) and _s.get("image") == frame["origin"]:          # zone déjà tracée sur cette image
        _x0, _y0, _x1, _y1 = (int(t) for t in _s["r"])
        _x0, _x1 = min(max(_x0, 0), W_img - 2), min(max(_x1, 1), W_img - 1)
        _y0, _y1 = min(max(_y0, 0), H_img - 2), min(max(_y1, 1), H_img - 1)
    if RectPicker is not None:
        rect = mo.ui.anywidget(RectPicker(image=png_uri(img), w=W_img, h=H_img, width=440,
                                          u0=int(_x0), v0=int(_y0), u1=int(_x1), v1=int(_y1)))
        roi_u0 = roi_v0 = roi_w = roi_h = None
    else:      # anywidget absent : 4 curseurs
        rect = None
        roi_u0 = mo.ui.slider(0, W_img - 1, value=int(_x0), label="ROI : u0", show_value=True)
        roi_v0 = mo.ui.slider(0, H_img - 1, value=int(_y0), label="ROI : v0", show_value=True)
        roi_w = mo.ui.slider(5, W_img, value=max(5, int(_x1 - _x0)), label="ROI : largeur", show_value=True)
        roi_h = mo.ui.slider(5, H_img, value=max(5, int(_y1 - _y0)), label="ROI : hauteur", show_value=True)
    return rect, roi_h, roi_u0, roi_v0, roi_w


@app.cell(hide_code=True)
def _(H_img, W_img, etat_set, frame, rect, roi_h, roi_u0, roi_v0, roi_w):
    # (u0, v0, u1, v1) : la région d'échantillon
    if rect is not None:
        _a = (int(rect.value["u0"]), int(rect.value["v0"]), int(rect.value["u1"]), int(rect.value["v1"]))
    else:
        _a = (roi_u0.value, roi_v0.value, roi_u0.value + roi_w.value, roi_v0.value + roi_h.value)
    roi = (max(0, min(_a[0], W_img - 1)), max(0, min(_a[1], H_img - 1)),
           max(1, min(_a[2], W_img)), max(1, min(_a[3], H_img)))
    etat_set("roi", {"image": frame["origin"], "r": [int(t) for t in roi]})
    return (roi,)


@app.cell(hide_code=True)
def _(
    block,
    cv2,
    hsv,
    img,
    mo,
    np,
    plt,
    rect,
    etat_set,
    roi,
    roi_h,
    roi_u0,
    roi_v0,
    roi_w,
    set_bounds,
    to_rgb,
):
    _u0, _v0, _u1, _v1 = roi
    _roi = hsv[_v0:_v1, _u0:_u1].reshape(-1, 3)
    _fig, _ax = plt.subplots(1, 3, figsize=(9.5, 2.5))
    for _i, (_name, _mx) in enumerate([("H (0-179)", 180), ("S (0-255)", 256), ("V (0-255)", 256)]):
        _cnt, _ = np.histogram(_roi[:, _i], bins=np.arange(_mx + 1))
        if _i == 0:
            _cols = [plt.matplotlib.colors.hsv_to_rgb((_k / 180, 0.7, 0.95)) for _k in range(180)]
            _ax[_i].bar(range(180), _cnt, width=1, color=_cols)
        else:
            _ax[_i].bar(range(_mx), _cnt, width=1, color="#5B7F95")
        _ax[_i].set_title(_name, fontsize=10)
        for _s in ("top", "right"):
            _ax[_i].spines[_s].set_visible(False)
    _fig.tight_layout()
    _p = lambda c, q: int(np.percentile(_roi[:, c], q)) if len(_roi) else 0
    _sugg = (_p(0, 2), _p(0, 98), _p(1, 2), _p(2, 2))        # H min, H max, S min, V min

    apply_btn = mo.ui.button(label="⬇️ Utiliser ces bornes", kind="success",
                             on_click=lambda _: (set_bounds(_sugg),
                                                 [etat_set(_k, int(_v)) for _k, _v in
                                                  zip(("r:h_min", "r:h_max", "r:s_min", "r:v_min"), _sugg)]))
    if rect is not None:
        _left = rect
    else:
        _view = img.copy()
        cv2.rectangle(_view, (_u0, _v0), (_u1, _v1), (255, 255, 255), 4)
        cv2.rectangle(_view, (_u0, _v0), (_u1, _v1), (0, 0, 0), 2)
        _left = mo.vstack([mo.callout(mo.md("`pip install anywidget` pour **tracer le rectangle à la souris**."), kind="warn"),
                           roi_u0, roi_v0, roi_w, roi_h, mo.image(to_rgb(_view), width=440)])
    block(
        "vous", "3.1 · Trouver les bornes",
        "*Tracez un rectangle **sur l'objet** : les histogrammes montrent où se concentrent ses teintes, saturations et valeurs.*",
        mo.hstack([
            _left,
            mo.vstack([
                _fig,
                mo.md(f"**Bornes suggérées** (2 % – 98 % des pixels de la région) : "
                      f"H ∈ [{_sugg[0]}, {_sugg[1]}] · S ≥ {_sugg[2]} · V ≥ {_sugg[3]}"),
                apply_btn,
                mo.md("*Laissez de la marge, puis ajustez à la main ci-dessous.*"),
            ], gap=0.6),
        ], justify="start", gap=1.5, align="start", wrap=True),
    )
    return


@app.cell(hide_code=True)
def _(block, etat_get, etat_set, get_bounds, mo):
    _hmin, _hmax, _smin, _vmin = get_bounds()
    _sv = lambda k: (lambda v: etat_set("r:" + k, int(v)))          # sauvegarde à chaque changement
    h_min = mo.ui.slider(0, 179, value=int(_hmin), label="H min", show_value=True, on_change=_sv("h_min"))
    h_max = mo.ui.slider(0, 179, value=int(_hmax), label="H max", show_value=True, on_change=_sv("h_max"))
    s_min = mo.ui.slider(0, 255, value=int(_smin), label="S min", show_value=True, on_change=_sv("s_min"))
    s_max = mo.ui.slider(0, 255, value=int(etat_get("r:s_max", 255)), label="S max", show_value=True, on_change=_sv("s_max"))
    v_min = mo.ui.slider(0, 255, value=int(_vmin), label="V min", show_value=True, on_change=_sv("v_min"))
    v_max = mo.ui.slider(0, 255, value=int(etat_get("r:v_max", 255)), label="V max", show_value=True, on_change=_sv("v_max"))
    mo.callout(mo.md("🎚️ Les **curseurs des bornes** (H, S, V) se trouvent juste au-dessus du résultat de **3.2**, pour voir l'effet en direct. "
                     "Ils servent de cadre à toute la suite."), kind="neutral")
    return h_max, h_min, s_max, s_min, v_max, v_min


@app.cell(hide_code=True)
def _(h_max, h_min, np, s_max, s_min, v_max, v_min):
    # les bornes réglées en 3.1, sous forme de tableaux [H, S, V]
    lo = np.array([h_min.value, s_min.value, v_min.value])
    hi = np.array([h_max.value, s_max.value, v_max.value])
    return hi, lo


@app.cell(hide_code=True)
def _(sn_block, sn_editor):
    sed_mask = sn_editor("mask")
    sn_block("mask", sed_mask)
    return (sed_mask,)


@app.cell(hide_code=True)
def _(hi, hsv, lo, sed_mask, sn_eval, sn_view):
    sev_mask = sn_eval("mask", sed_mask.value, hsv=hsv, lo=lo, hi=hi)
    mask_raw = sev_mask["values"]["mask"]
    mask_ok = sev_mask["good"]
    sn_view(sev_mask)
    return mask_ok, mask_raw


@app.cell(hide_code=True)
def _(sn_block, sn_editor):
    sed_objet = sn_editor("objet")
    sn_block("objet", sed_objet)
    return (sed_objet,)


@app.cell(hide_code=True)
def _(img, mask_raw, sed_objet, sn_eval, sn_view):
    sev_objet = sn_eval("objet", sed_objet.value, img=img, mask=mask_raw)
    objet = sev_objet["values"]["objet"]
    objet_ok = sev_objet["good"]
    sn_view(sev_objet)
    return objet, objet_ok


@app.cell(hide_code=True)
def _(
    block,
    cv2,
    figure,
    fr,
    h_max,
    h_min,
    hsv,
    hue_strip,
    img,
    mask_ok,
    mask_raw,
    mo,
    objet,
    objet_ok,
    pu,
    pv,
    s_max,
    s_min,
    to_rgb,
    v_max,
    v_min,
):
    mo.stop(not (mask_ok and objet_ok), mo.callout(mo.md("🔒 *Les résultats s'afficheront dès que **3.2a** et **3.2b** seront validés.*"), kind="neutral"))
    _view = img.copy()
    cv2.circle(_view, (pu, pv), 12, (0, 0, 255), 2)
    _h, _s, _v = (int(x) for x in hsv[pv, pu])
    _t = lambda x, a, b: "✅" if a <= x <= b else "❌"
    block(
        "vous", "3.2 · Le résultat de votre seuillage",
        mo.md("**Bornes du seuillage** *(le cadre pour toute la suite)*"),
        mo.hstack([
            mo.vstack([mo.md("**Teinte**"), h_min, h_max]),
            mo.vstack([mo.md("**Saturation**"), s_min, s_max]),
            mo.vstack([mo.md("**Valeur**"), v_min, v_max]),
        ], justify="start", gap=2, wrap=True),
        mo.hstack([
            figure(to_rgb(_view), "*Image + pixel choisi*"),
            figure(mask_raw, "*Votre masque* (blanc = 255)"),
            figure(to_rgb(objet), "*Votre objet isolé*"),
        ], justify="start", gap=1, wrap=True),
        f"Pixels gardés : **{cv2.countNonZero(mask_raw)}** ({fr(100 * cv2.countNonZero(mask_raw) / mask_raw.size, 1)} %)",
        mo.ui.tabs({
            "🎯 Test sur le pixel choisi": mo.md(f"""
    `H, S, V = ({_h}, {_s}, {_v})` au pixel ({pu}, {pv})

    | canal | bas ≤ valeur ≤ haut | test |
    |---|---|---|
    | H | {h_min.value} ≤ **{_h}** ≤ {h_max.value} | {_t(_h, h_min.value, h_max.value)} |
    | S | {s_min.value} ≤ **{_s}** ≤ {s_max.value} | {_t(_s, s_min.value, s_max.value)} |
    | V | {v_min.value} ≤ **{_v}** ≤ {v_max.value} | {_t(_v, v_min.value, v_max.value)} |

    → `mask[{pv}, {pu}] = {int(mask_raw[pv, pu])}`
                """),
            "🌈 Teintes retenues": mo.image(to_rgb(hue_strip([(h_min.value, h_max.value)])), width=700),
        }),
    )
    return


@app.cell(hide_code=True)
def _(block, etat_get, etat_set, mo):
    k_open = mo.ui.slider(1, 31, step=2, value=int(etat_get("r:k_open", 7)), label="Taille du motif d'ouverture (1 = aucune)",
                          show_value=True, on_change=lambda v: etat_set("r:k_open", int(v)))
    k_close = mo.ui.slider(1, 41, step=2, value=int(etat_get("r:k_close", 15)), label="Taille du motif de fermeture (1 = aucune)",
                           show_value=True, on_change=lambda v: etat_set("r:k_close", int(v)))
    mo.vstack([
        mo.md("### 3.3 · Nettoyer le masque : morphologie (slides 18-19)"),
        block("rappel", "deux opérations sur le masque",
              "**Ouverture** = érosion puis dilatation : les petits objets blancs (bruit) disparaissent. "
              "**Fermeture** = dilatation puis érosion : les petits trous noirs (reflets, ombres) sont comblés. "
              "Le **motif** (élément structurant) fixe la taille des défauts concernés."),
    ])
    return k_close, k_open


@app.cell(hide_code=True)
def _(sn_block, sn_editor):
    sed_open = sn_editor("open")
    sn_block("open", sed_open)
    return (sed_open,)


@app.cell(hide_code=True)
def _(k_open, mask_raw, sed_open, sn_eval, sn_view):
    sev_open = sn_eval("open", sed_open.value, mask=mask_raw, k_open=k_open.value)
    mask_open = sev_open["values"]["opened"]
    open_ok = sev_open["good"]
    sn_view(sev_open)
    return mask_open, open_ok


@app.cell(hide_code=True)
def _(sn_block, sn_editor):
    sed_close = sn_editor("close")
    sn_block("close", sed_close)
    return (sed_close,)


@app.cell(hide_code=True)
def _(k_close, mask_open, sed_close, sn_eval, sn_view):
    sev_close = sn_eval("close", sed_close.value, opened=mask_open, k_close=k_close.value)
    mask_clean = sev_close["values"]["closed"]
    close_ok = sev_close["good"]
    sn_view(sev_close)
    return close_ok, mask_clean


@app.cell(hide_code=True)
def _(
    block,
    close_ok,
    cv2,
    figure,
    k_close,
    k_open,
    mask_clean,
    mask_open,
    mask_raw,
    mo,
    open_ok,
):
    mo.stop(not (open_ok and close_ok), mo.callout(mo.md("🔒 *Les résultats s'afficheront dès que **3.3a** et **3.3b** seront validés.*"), kind="neutral"))
    _n = lambda m: cv2.connectedComponents(m)[0] - 1
    # démonstration du piège : un motif énorme
    _big = cv2.morphologyEx(mask_raw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61)))
    _pct = lambda m: 100 * cv2.countNonZero(m) / max(1, cv2.countNonZero(mask_raw))
    mo.ui.tabs({
        "① Votre ouverture : supprime le bruit": block(
            "vous", "avant / après l'ouverture",
            f"Motif de {k_open.value} px. Un petit objet blanc disparaît s'il ne contient pas tout le motif.",
            k_open,
            mo.hstack([
                figure(mask_raw, f"*Masque brut* — **{_n(mask_raw)}** régions"),
                figure(mask_open, f"*Après votre ouverture* — **{_n(mask_open)}** régions"),
            ], justify="start", gap=1, wrap=True),
        ),
        "② Votre fermeture : bouche les trous": block(
            "vous", "avant / après la fermeture",
            f"Motif de {k_close.value} px. Les petits trous noirs dans l'objet sont comblés.",
            k_close,
            mo.hstack([
                figure(mask_open, f"*Après ouverture* — **{_n(mask_open)}** régions"),
                figure(mask_clean, f"*Après votre fermeture* — **{_n(mask_clean)}** régions"),
            ], justify="start", gap=1, wrap=True),
        ),
        "⚠️ Piège : un motif trop grand": block(
            "piege", "la taille du motif ≈ la taille du défaut à effacer",
            "Avec un motif d'ouverture de 61 × 61, l'objet lui-même est abîmé :",
            mo.hstack([
                figure(mask_raw, "*Masque brut*"),
                figure(_big, f"*Ouverture 61 × 61* — il reste **{_pct(_big):.0f} %** des pixels"),
            ], justify="start", gap=1, wrap=True),
        ),
    })
    return


@app.cell(hide_code=True)
def _(block, mo):
    mo.vstack([
        mo.md("### 3.4 · De vos lignes à des fonctions réutilisables"),
        block("rappel", "des variables du notebook aux paramètres",
              "Jusqu'ici votre code utilisait des variables du notebook (`hsv`, `lo`, `k_open`…). "
              "`vision_node.py` ne les a pas : il lui faut des **fonctions** dont tout ce qu'elles utilisent est passé en **paramètres**. "
              "Une fois validées, elles sont exportées dans votre module `ma_vision.py`."),
    ])
    return


@app.cell(hide_code=True)
def _(ex_block, ex_editor):
    ed_hsv = ex_editor("hsv_mask")
    ex_block("hsv_mask", ed_hsv)
    return (ed_hsv,)


@app.cell(hide_code=True)
def _(ed_hsv, evaluate, ex_view, hi, img, lo, mask_raw, np):
    ev_hsv = evaluate("hsv_mask", ed_hsv.value)
    _extra = None
    if ev_hsv["fn"] is not None and ev_hsv["good"]:
        try:
            _same = np.array_equal(ev_hsv["fn"](img, lo, hi), mask_raw)
            _extra = "Sur **votre image**, avec vos bornes : " + ("✅ même masque que celui de la section 3." if _same else "❌ masque différent de celui de la section 3.")
        except Exception as _e:
            _extra = f"Sur votre image : ❌ {type(_e).__name__} : {_e}"
    ex_view(ev_hsv, _extra)
    return (ev_hsv,)


@app.cell(hide_code=True)
def _(ex_block, ex_editor):
    ed_clean = ex_editor("clean_mask")
    ex_block("clean_mask", ed_clean)
    return (ed_clean,)


@app.cell(hide_code=True)
def _(ed_clean, evaluate, ex_view, k_close, k_open, mask_clean, mask_raw, np):
    ev_clean = evaluate("clean_mask", ed_clean.value)
    _extra = None
    if ev_clean["fn"] is not None and ev_clean["good"]:
        try:
            _same = np.array_equal(ev_clean["fn"](mask_raw, k_open.value, k_close.value), mask_clean)
            _extra = "Sur **votre masque**, avec vos motifs : " + ("✅ même résultat que la section 3.3." if _same else "❌ résultat différent de la section 3.3.")
        except Exception as _e:
            _extra = f"Sur votre masque : ❌ {type(_e).__name__} : {_e}"
    ex_view(ev_clean, _extra)
    return (ev_clean,)


@app.cell(hide_code=True)
def _(block, h_max, h_min, k_close, k_open, s_max, s_min, v_max, v_min):
    block(
        "retenir", "section 3 : vos réglages à reporter dans `vision_node.py`",
        f"""
    | | min | max |
    |---|---|---|
    | **H** | {h_min.value} | {h_max.value} |
    | **S** | {s_min.value} | {s_max.value} |
    | **V** | {v_min.value} | {v_max.value} |

    Motifs : ouverture **{k_open.value}** px · fermeture **{k_close.value}** px.

    - Le masque est un tableau de **0 / 255** (`uint8`), de la même taille que l'image.
    - L'ouverture supprime le bruit, la fermeture bouche les trous ; la taille du motif ≈ la taille du défaut.
            """,
    )
    return


@app.cell(hide_code=True)
def _(block, mo):
    mo.vstack([
        mo.md("---\n# 4 · Contour"),
        block("rappel", "la frontière d'une région",
              """
    Un **contour** est la **liste ordonnée des pixels de la frontière** d'une région du masque. Chaque contour est un tableau `(N, 1, 2)` de points `(u, v)`.

    Dans l'application finale on veut la position de **tous** les objets de la scène : on garde donc **chaque** contour assez grand, pas seulement le plus grand.

    Trois étapes, **à écrire vous-même** : **4.1** trouver les contours · **4.2** garder les objets · **4.3** en faire une fonction.
    Une mise en garde sur **Canny** conclut la section.
                  """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    rt_mode = mo.ui.dropdown({"RETR_EXTERNAL (extérieurs seulement)": "RETR_EXTERNAL", "RETR_LIST (tous, à plat)": "RETR_LIST",
                              "RETR_CCOMP (extérieurs + trous)": "RETR_CCOMP", "RETR_TREE (hiérarchie complète)": "RETR_TREE"},
                             value="RETR_EXTERNAL (extérieurs seulement)", label="Mode")
    rt_method = mo.ui.dropdown({"CHAIN_APPROX_NONE (tous les points)": "CHAIN_APPROX_NONE",
                                "CHAIN_APPROX_SIMPLE (extrémités des segments)": "CHAIN_APPROX_SIMPLE"},
                               value="CHAIN_APPROX_SIMPLE (extrémités des segments)", label="Méthode")
    return rt_method, rt_mode


@app.cell(hide_code=True)
def _(block, contours, contours_ok, cv2, mo, np, rt_method, rt_mode):
    mo.stop(not contours_ok, mo.callout(mo.md("🔒 *La suite s'affichera dès que **4.1** sera validé.*"), kind="neutral"))
    # ---- un masque d'exemple où les réglages font une différence : un anneau, un objet dans le trou, une autre région
    _ex = np.zeros((220, 320), np.uint8)
    cv2.ellipse(_ex, (110, 110), (90, 80), 0, 0, 360, 255, -1)
    cv2.circle(_ex, (110, 110), 38, 0, -1)
    cv2.circle(_ex, (110, 110), 14, 255, -1)
    cv2.rectangle(_ex, (230, 60), (290, 150), 255, -1)
    _cs, _hier = cv2.findContours(_ex, getattr(cv2, rt_mode.value), getattr(cv2, rt_method.value))
    _ov = cv2.cvtColor(_ex // 3, cv2.COLOR_GRAY2BGR)
    _cols = [(30, 30, 230), (230, 120, 20), (40, 160, 40), (200, 40, 200), (0, 200, 220), (200, 200, 0)]
    for _i, _c in enumerate(_cs):
        cv2.drawContours(_ov, [_c], -1, _cols[_i % 6], 2)
    _pts = sum(len(_c) for _c in _cs)
    _holes = sum(1 for _h in (_hier[0] if _hier is not None else []) if _h[3] >= 0)
    _mine = f"Sur **votre** masque, votre appel a trouvé **{len(contours)}** contour(s) ({sum(len(c) for c in contours)} points)."
    block(
        "vous", "4.1 · Les deux réglages de `findContours` (slide 23)",
        "*Votre masque n'a ni trou ni objet emboîté : les modes y donnent le même résultat. Pour voir la différence, essayez-les sur ce masque d'exemple (un anneau, un petit objet dans son trou, une autre région).*",
        mo.hstack([rt_mode, rt_method], justify="start", wrap=True),
        mo.hstack([
            mo.image(cv2.cvtColor(_ov, cv2.COLOR_BGR2RGB), width=480),
            mo.md(f"""
    **{len(_cs)} contour(s)** · **{_pts} points** au total

    dont **{_holes}** contour(s) de trou

    *Avec « extérieurs seulement » : le trou et l'objet qu'il contient disparaissent, il reste 2 contours. Avec « extrémités des segments » : peu de points. C'est ce qu'on veut ici.*
                """),
        ], justify="start", gap=1.5, align="center", wrap=True),
        _mine,
    )
    return


@app.cell(hide_code=True)
def _(sn_block, sn_editor):
    sed_contours = sn_editor("contours")
    sn_block("contours", sed_contours)
    return (sed_contours,)


@app.cell(hide_code=True)
def _(mask_clean, sed_contours, sn_eval, sn_view):
    sev_contours = sn_eval("contours", sed_contours.value, mask_clean=mask_clean)
    contours = sev_contours["values"]["contours"]
    contours_ok = sev_contours["good"]
    sn_view(sev_contours)
    return contours, contours_ok


@app.cell(hide_code=True)
def _(block, etat_get, etat_set, mo):
    min_area = mo.ui.slider(0, 20000, step=50, value=int(etat_get("r:min_area", 500)), label="Aire minimale d'un objet (px²)",
                            show_value=True, on_change=lambda v: etat_set("r:min_area", int(v)))
    return (min_area,)


@app.cell(hide_code=True)
def _(sn_block, sn_editor):
    sed_objects = sn_editor("objects")
    sn_block("objects", sed_objects)
    return (sed_objects,)


@app.cell(hide_code=True)
def _(contours, min_area, sed_objects, sn_eval, sn_view):
    sev_objects = sn_eval("objects", sed_objects.value, contours=contours, min_area=int(min_area.value))
    objects = list(sev_objects["values"]["objects"])
    objects_ok = sev_objects["good"]
    sn_view(sev_objects)
    return objects, objects_ok


@app.cell(hide_code=True)
def _(mo, objects):
    obj_sel = mo.ui.number(1, max(1, len(objects)), value=1, step=1, label="Objet inspecté (n°)")
    return (obj_sel,)


@app.cell(hide_code=True)
def _(obj_sel, objects):
    # l'objet « inspecté » : les sections 5 et 6 détaillent un objet à la fois (le n° 1 = le plus grand, par défaut)
    c_obj = objects[min(int(obj_sel.value), len(objects)) - 1] if len(objects) else None
    obj_found = c_obj is not None
    return c_obj, obj_found


@app.cell(hide_code=True)
def _(
    block,
    c_obj,
    contours,
    cv2,
    img,
    mask_clean,
    min_area,
    mo,
    np,
    obj_found,
    obj_sel,
    objects,
    objects_ok,
    to_rgb,
):
    mo.stop(not objects_ok, mo.callout(mo.md("🔒 *Les résultats s'afficheront dès que **4.2** sera validé.*"), kind="neutral"))
    _cs = sorted(contours, key=cv2.contourArea, reverse=True)
    _idx = lambda c: next((k for k, o in enumerate(objects) if o.shape == c.shape and np.array_equal(o, c)), None)
    _sel = None if c_obj is None else int(min(int(obj_sel.value), len(objects))) - 1
    _ov = img.copy()
    _rows = ["| n° | aire `contourArea` (px²) | points | boîte (u, v, l, h) | ≥ seuil | objet n° |", "|---|---|---|---|---|---|"]
    for _c in _cs:
        _k = _idx(_c)
        _col = (40, 200, 40) if _k is not None else (60, 60, 230)          # BGR : vert = objet, rouge = trop petit
        cv2.drawContours(_ov, [_c], -1, _col, 5 if _k == _sel else 2)
        _x, _y, _w, _h = cv2.boundingRect(_c)
        if _k is not None:
            cv2.putText(_ov, str(_k + 1), (_x + 4, _y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 5)
            cv2.putText(_ov, str(_k + 1), (_x + 4, _y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 130, 30), 2)
    for _n, _c in enumerate(_cs[:8], 1):
        _k = _idx(_c)
        _x, _y, _w, _h = cv2.boundingRect(_c)
        _rows.append(f"| {_n} | {cv2.contourArea(_c):.0f} | {len(_c)} | ({_x}, {_y}, {_w}, {_h}) | {'✅' if cv2.contourArea(_c) >= min_area.value else '❌'} | {'**' + str(_k + 1) + '**' if _k is not None else '—'} |")

    if c_obj is not None:
        _p = c_obj[:, 0, :].astype(np.float64)
        _lacet = 0.5 * abs(np.sum(_p[:, 0] * np.roll(_p[:, 1], -1) - np.roll(_p[:, 0], -1) * _p[:, 1]))
        _fill = np.zeros_like(mask_clean)
        cv2.drawContours(_fill, [c_obj], -1, 255, cv2.FILLED)
        _lacet_md = (f"Objet n° {_sel + 1}, écrite en NumPy : **{_lacet:.1f} px²** · `cv2.contourArea` : **{cv2.contourArea(c_obj):.1f} px²** "
                     f"{'✅ identique' if abs(_lacet - cv2.contourArea(c_obj)) < 0.6 else '⚠️ écart'}.\n\n"
                     f"Pixels de la région remplie (`cv2.countNonZero`) : **{cv2.countNonZero(_fill)}** → l'aire du contour est un peu plus "
                     f"**petite** : le polygone passe par les *centres* des pixels de bord.")
    else:
        _lacet_md = "Pas d'objet retenu."

    mo.vstack([
        block(
            "vous", "4.2 · Vos objets (slide 24)",
            f"Votre code a gardé **{len(objects)} objet(s)**. *Vert : objet gardé (avec son n°) · rouge : sous le seuil. Un contour plus petit que le seuil est du bruit, pas un objet : déplacez le curseur, votre code est réévalué.*",
            min_area,
            mo.hstack([
                mo.image(to_rgb(_ov), width=480),
                mo.md("\n".join(_rows) + ("" if obj_found else "\n\n> ⚠️ **Aucun objet** : élargissez les bornes de la section 3 ou baissez l'aire minimale.")),
            ], justify="start", gap=1.5, align="center", wrap=True),
            "Les sections 5 et 6 détaillent **un objet à la fois** ; dans l'application, la même fonction sera appliquée à chacun. Choisissez celui à inspecter :",
            obj_sel,
        ),
        mo.ui.tabs({
            "🧮 La formule du lacet": block(
                "rappel", "l'aire d'un contour",
                r"$A = \tfrac12 \left| \sum_i (x_i\,y_{i+1} - x_{i+1}\,y_i) \right|$",
                _lacet_md,
            ),
            "⚠️ Piège : l'aire dépend de la distance": block(
                "piege", "un seuil d'aire n'est pas universel",
                "L'aire apparente d'un objet varie en **1/Z²** : deux fois plus loin, quatre fois moins de pixels.",
                "Un seuil réglé de près peut **éliminer l'objet** quand on recule. Gardez-le bas, ou adaptez-le à la plage de travail.",
            ),
        }),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    canny_lo = mo.ui.slider(0, 255, value=50, label="Canny : seuil bas", show_value=True)
    canny_hi = mo.ui.slider(0, 255, value=130, label="Canny : seuil haut", show_value=True)
    return canny_hi, canny_lo


@app.cell(hide_code=True)
def _(
    block,
    canny_hi,
    canny_lo,
    contours_ok,
    cv2,
    figure,
    img,
    mask_clean,
    mo,
    to_rgb,
):
    mo.stop(not contours_ok)
    _gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _edges = cv2.Canny(cv2.GaussianBlur(_gray, (5, 5), 0), canny_lo.value, canny_hi.value)
    _cs, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    _ov = img.copy()
    cv2.drawContours(_ov, _cs, -1, (255, 255, 255), 5)
    cv2.drawContours(_ov, _cs, -1, (30, 30, 30), 2)
    mo.accordion({
        "⚠️ Ne pas confondre avec les « contours » de Canny": block(
            "piege", "deux notions derrière le mot « contour »",
            "**Canny** détecte les pixels où l'image **change vite** (gradient) : un nuage de points, sans ordre ni regroupement, sensible à la texture. "
            "**`findContours`** suit la **frontière fermée d'une région du masque** : c'est elle dont on tire l'aire et le centroïde.",
            mo.hstack([canny_lo, canny_hi], justify="start", wrap=True),
            mo.hstack([
                figure(255 - _edges, "*Canny : pixels de bord (gradient)*", 440),
                figure(to_rgb(_ov), "*`findContours` : courbes fermées autour des régions du masque*", 440),
            ], justify="start", gap=1.5, wrap=True),
        ),
    })
    return


@app.cell(hide_code=True)
def _(ex_block, ex_editor):
    ed_fo = ex_editor("find_objects")
    ex_block("find_objects", ed_fo)
    return (ed_fo,)


@app.cell(hide_code=True)
def _(ed_fo, evaluate, ex_view, mask_clean, min_area, np, objects):
    ev_fo = evaluate("find_objects", ed_fo.value)
    _extra = None
    if ev_fo["fn"] is not None and ev_fo["good"]:
        try:
            _r = list(ev_fo["fn"](mask_clean, int(min_area.value)))
            _same = len(_r) == len(objects) and all(np.array_equal(a, b) for a, b in zip(_r, objects))
            _extra = f"Sur **votre masque**, avec votre seuil : {len(_r)} objet(s) · " + ("✅ mêmes objets qu'en 4.2." if _same else "❌ résultat différent de celui de 4.2.")
        except Exception as _e:
            _extra = f"Sur votre masque : ❌ {type(_e).__name__} : {_e}"
    ex_view(ev_fo, _extra)
    return (ev_fo,)


@app.cell(hide_code=True)
def _(block, min_area, objects):
    block(
        "retenir", "section 4",
        f"""
    - Un contour est un tableau `(N, 1, 2)` de points `(u, v)` ; `findContours` travaille sur un masque **binaire** et renvoie **deux** valeurs : les contours et leur hiérarchie.
    - Contours **extérieurs** seulement (les trous et objets emboîtés sont ignorés), **peu de points** : c'est le choix du TP.
    - Les **objets** sont tous les contours d'aire ≥ un seuil (ici **{min_area.value} px²**), triés du plus grand au plus petit : ici **{len(objects)}** objet(s).
    - Aucun objet est un cas **normal** (liste vide) : l'objet est sorti du champ, on ne publie rien.
    - Le seuil d'aire dépend de la distance (aire ∝ 1/Z²) : gardez-le bas.
            """,
    )
    return


@app.cell(hide_code=True)
def _(block, mo):
    mo.vstack([
        mo.md("---\n# 5 · Centroïde"),
        block("rappel", "le barycentre de la région",
              r"""
    Le **centroïde** est la **moyenne des positions** des pixels de la région. Avec les **moments** $m_{pq}$ de la région (des sommes de $u^p\, v^q$) :

    $$m_{00} = \text{aire}, \qquad u_c = \frac{m_{10}}{m_{00}}, \qquad v_c = \frac{m_{01}}{m_{00}}$$

    OpenCV sait calculer ces moments directement **à partir du contour** de l'objet : pas besoin de masque.
    Trois étapes à écrire : **5.1** les moments · **5.2** le centroïde · **5.3** la fonction.
    Ces sections détaillent l'objet n° choisi en 4.2 ; la fonction servira pour **chaque** objet.
                  """),
    ])
    return


@app.cell(hide_code=True)
def _(sn_block, sn_editor):
    sed_moments = sn_editor("moments")
    sn_block("moments", sed_moments)
    return (sed_moments,)


@app.cell(hide_code=True)
def _(c_obj, mo, sed_moments, sn_eval, sn_view):
    if c_obj is None:
        M = None
        moments_ok = False
        _out = mo.callout(mo.md("⚠️ **Aucun objet** (section 4) : élargissez les bornes de la section 3 ou baissez l'aire minimale."), kind="warn")
    else:
        _sev = sn_eval("moments", sed_moments.value, c_obj=c_obj)
        M = _sev["values"]["M"]
        moments_ok = _sev["good"]
        _out = sn_view(_sev)
    _out
    return M, moments_ok


@app.cell(hide_code=True)
def _(sn_block, sn_editor):
    sed_centre = sn_editor("centre")
    sn_block("centre", sed_centre)
    return (sed_centre,)


@app.cell(hide_code=True)
def _(M, sed_centre, sn_eval, sn_view):
    if M is None:
        u_c = v_c = u_i = v_i = None
        centre_ok = False
        _out = None
    else:
        _sev = sn_eval("centre", sed_centre.value, M=M)
        u_c, v_c = _sev["values"]["u_c"], _sev["values"]["v_c"]
        centre_ok = _sev["good"]
        u_i, v_i = (int(round(u_c)), int(round(v_c))) if u_c is not None else (None, None)   # pour indexer / dessiner : entiers
        _out = sn_view(_sev)
    _out
    return centre_ok, u_c, u_i, v_c, v_i


@app.cell(hide_code=True)
def _(
    M,
    block,
    c_obj,
    centre_ok,
    cv2,
    fr,
    img,
    mask_clean,
    mo,
    moments_ok,
    np,
    objects,
    to_rgb,
    u_c,
    u_i,
    v_c,
    v_i,
):
    mo.stop(c_obj is None)
    mo.stop(not (moments_ok and centre_ok), mo.callout(mo.md("🔒 *Les résultats s'afficheront dès que **5.1** et **5.2** seront validés.*"), kind="neutral"))
    # ---- les mêmes calculs, de plusieurs façons
    _fill = np.zeros_like(mask_clean)
    cv2.drawContours(_fill, [c_obj], -1, 255, cv2.FILLED)          # région remplie
    _holes = cv2.bitwise_and(mask_clean, _fill)                    # région avec ses trous (reflets)
    _Mf = cv2.moments(_fill, binaryImage=True)
    _Mh = cv2.moments(_holes, binaryImage=True)
    _M255 = cv2.moments(_fill)                                     # sans binaryImage : les 255 comptent
    _ys, _xs = np.nonzero(_fill)                                   # « moyenne des positions » à la main
    _bx, _by, _bw, _bh = cv2.boundingRect(c_obj)
    (_ex, _ey), _er = cv2.minEnclosingCircle(c_obj)
    _pb, _pe = (_bx + _bw / 2, _by + _bh / 2), (_ex, _ey)

    _view = img.copy()
    for _o in objects:                                             # tous les objets : un point par centroïde
        _Mo = cv2.moments(_o)
        if _Mo["m00"] > 0:
            _q = (int(round(_Mo["m10"] / _Mo["m00"])), int(round(_Mo["m01"] / _Mo["m00"])))
            cv2.drawContours(_view, [_o], -1, (255, 255, 255), 2)
            cv2.circle(_view, _q, 8, (255, 255, 255), -1)
            cv2.circle(_view, _q, 5, (40, 160, 40), -1)
    cv2.drawContours(_view, [c_obj], -1, (255, 255, 255), 3)
    cv2.rectangle(_view, (_bx, _by), (_bx + _bw, _by + _bh), (58, 131, 232), 2)
    for (_p, _col, _r) in [(_pb, (58, 131, 232), 7), (_pe, (40, 160, 40), 7), ((u_c, v_c), (30, 30, 230), 10)]:
        _q = (int(round(_p[0])), int(round(_p[1])))
        cv2.circle(_view, _q, _r + 3, (255, 255, 255), -1)
        cv2.circle(_view, _q, _r, _col, -1)
    _d = lambda p: float(np.hypot(p[0] - u_c, p[1] - v_c))

    mo.vstack([
        block(
            "vous", "5.2 · Le centroïde de votre objet",
            mo.hstack([
                mo.image(to_rgb(_view), width=520),
                mo.md(f"""
    `m00 = {M['m00']:.0f}` (aire du contour) · `m10 = {M['m10']:.0f}` · `m01 = {M['m01']:.0f}`

    ### (u<sub>c</sub>, v<sub>c</sub>) = ({fr(u_c, 2)} ; {fr(v_c, 2)})

    Valeurs décimales : **précision sous-pixel**. Pour lire un tableau : `[v, u] = [{v_i}, {u_i}]`

    🔴 centroïde de l'objet inspecté · 🟠 centre de la boîte `boundingRect` · 🟢 centre du cercle `minEnclosingCircle` ; les petits points verts sont les centroïdes des **autres** objets.
                    """),
            ], justify="start", gap=1.5, align="center", wrap=True),
        ),
        mo.ui.tabs({
            "🔍 Contour, région ou masque ?": block(
                "vous", "quatre façons de calculer le même centroïde",
                "Le masque a des **trous** (reflets) : ils comptent dans un calcul, pas dans l'autre. Le contour, lui, n'a pas de trous.",
                mo.md(f"""
    | Calcul sur… | m00 | u<sub>c</sub> | v<sub>c</sub> |
    |---|---|---|---|
    | le contour — **le vôtre** | {M['m00']:.0f} | {fr(u_c, 3)} | {fr(v_c, 3)} |
    | la région remplie (pixels) | {_Mf['m00']:.0f} | {fr(_Mf['m10'] / _Mf['m00'], 3)} | {fr(_Mf['m01'] / _Mf['m00'], 3)} |
    | la région avec ses trous (`mask_clean`) | {_Mh['m00']:.0f} | {fr(_Mh['m10'] / _Mh['m00'], 3)} | {fr(_Mh['m01'] / _Mh['m00'], 3)} |
    | « moyenne des positions » (`np.nonzero`) | {len(_xs)} | {fr(_xs.mean(), 3)} | {fr(_ys.mean(), 3)} |
                    """),
                "*La dernière ligne est la définition même : la moyenne des `u` et des `v` des pixels. `m00` du contour est un peu plus petit que le nombre de pixels (le polygone passe par les centres des pixels de bord).*",
            ),
            "⚠️ Piège : moments d'un masque": block(
                "piege", "sur un masque, `binaryImage=True`",
                f"Si on calcule les moments d'un **masque** (0/255) au lieu d'un contour, sans `binaryImage=True`, `m00 = {_M255['m00']:.0f}` = 255 × {_Mf['m00']:.0f} pixels.",
                "Le centroïde reste juste (le facteur 255 s'annule dans le rapport) mais **l'aire est fausse**.",
                "Et toujours tester `m00 > 0` avant de diviser : un contour réduit à un point (aire nulle) plante sinon.",
            ),
            "📐 Pourquoi pas le centre de la boîte ?": block(
                "vous", "centroïde, centre de boîte, centre de cercle",
                mo.md(f"""
    | Point | Écart au centroïde |
    |---|---|
    | centre de la boîte `boundingRect` | {fr(_d(_pb), 1)} px |
    | centre du cercle `minEnclosingCircle` | {fr(_d(_pe), 1)} px |
                    """),
                "Le centroïde est la **moyenne de la masse** de l'objet. Le centre de la boîte ne dépend que des **extrémités** : un seul bout d'objet qui dépasse (une patte, une ombre) le déplace.",
            ),
        }),
    ])
    return


@app.cell(hide_code=True)
def _(ex_block, ex_editor):
    ed_cen = ex_editor("centroid")
    ex_block("centroid", ed_cen)
    return (ed_cen,)


@app.cell(hide_code=True)
def _(c_obj, ed_cen, evaluate, ex_view, np, u_c, v_c):
    ev_cen = evaluate("centroid", ed_cen.value)
    _extra = None
    if ev_cen["fn"] is not None and ev_cen["good"] and c_obj is not None and u_c is not None:
        try:
            _r = ev_cen["fn"](c_obj)
            _same = _r is not None and np.allclose(_r, (u_c, v_c), atol=1e-6)
            _extra = "Sur **votre objet inspecté** : " + ("✅ même centroïde qu'en 5.2." if _same else "❌ résultat différent de celui de 5.2.")
        except Exception as _e:
            _extra = f"Sur votre objet : ❌ {type(_e).__name__} : {_e}"
    ex_view(ev_cen, _extra)
    return (ev_cen,)


@app.cell(hide_code=True)
def _(block, fr, u_c, v_c):
    if u_c is None:
        _out = None
    else:
        _out = block(
            "retenir", "section 5",
            f"""
    - Le centroïde est la **moyenne des positions** des pixels de l'objet : `m10 / m00` (colonne `u_c`) et `m01 / m00` (ligne `v_c`) ; ici ({fr(u_c, 1)}, {fr(v_c, 1)}).
    - OpenCV calcule les moments **depuis le contour** ; `m00` est alors l'aire du contour. Sur un **masque**, il faudrait `binaryImage=True` (sinon `m00` = 255 × l'aire).
    - **Toujours tester `m00 > 0`** avant de diviser : un contour dégénéré (un point) n'a pas de centroïde, on renvoie `None`.
    - Le centroïde est **décimal** (sous-pixel) ; pour indexer un tableau il faut l'arrondir, et indexer **`[v, u]`**.
    - On le calcule pour **chaque** objet ; avec la profondeur, il donnera la position 3D de chacun (section 6).
                """,
        )
    _out
    return


@app.cell(hide_code=True)
def _(block, mo):
    mo.vstack([
        mo.md("---\n# 6 · Profondeur"),
        block("rappel", "un pixel n'est qu'un rayon",
              r"""
    La projection perd Z : un pixel correspond à tout un **rayon**. Pour retrouver le point 3D il faut **mesurer Z**, c'est le rôle de la caméra stéréo.
    Deux caméras voient le même point ; l'écart horizontal (**disparité** $d = u_l - u_r$) donne la profondeur :

    $$Z = \frac{b\,f}{d}, \qquad \Delta Z \approx \frac{Z^2}{b\,f}\,\Delta d$$

    La D405 est une caméra **stéréo passive** (pas de projecteur infrarouge) : une surface sans texture donne des trous (valeur 0).
    Cinq étapes : **6.1** la stéréo · **6.2** lire un pixel · **6.3** la profondeur de l'objet (son masque, puis la médiane) · **6.4** le point 3D · **6.5** les fonctions.
                  """),
    ])
    return


@app.cell(hide_code=True)
def _(frame):
    depth = frame["depth"]
    depth_scale = frame["depth_scale"]
    K = frame["K"]
    dist_coeffs = frame["dist"]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    has_depth = depth is not None
    return K, cx, cy, depth, depth_scale, dist_coeffs, fx, fy, has_depth


@app.cell(hide_code=True)
def _(fx, mo):
    st_b = mo.ui.slider(5, 120, value=18, label="Baseline b (mm)", show_value=True)
    st_f = mo.ui.slider(200, 1000, value=int(round(fx)), label="Focale f (pixels)", show_value=True)
    st_z = mo.ui.slider(7, 60, value=30, label="Distance Z (cm)", show_value=True)
    st_dd = mo.ui.slider(0.02, 1.0, step=0.02, value=0.1, label="Erreur de disparité Δd (pixels)", show_value=True)
    return st_b, st_dd, st_f, st_z


@app.cell(hide_code=True)
def _(block, fr, mo, np, plt, st_b, st_dd, st_f, st_z):
    _b, _f = st_b.value / 1000.0, float(st_f.value)       # b en mètres, f en pixels
    _Z = st_z.value / 100.0
    _d = _b * _f / _Z                                      # disparité en pixels
    _dZ = _Z ** 2 / (_b * _f) * st_dd.value                # erreur en mètres
    _zz = np.linspace(0.07, 0.6, 200)
    _fig, _ax = plt.subplots(figsize=(5.2, 3.0))
    _ax.plot(_zz * 100, _zz ** 2 / (_b * _f) * st_dd.value * 1000, color="#5B7F95", lw=2.5)
    _ax.plot(_Z * 100, _dZ * 1000, "o", color="#E8833A", ms=9)
    _ax.set_xlabel("distance Z (cm)"); _ax.set_ylabel("erreur ΔZ (mm)")
    for _s in ("top", "right"):
        _ax.spines[_s].set_visible(False)
    _fig.tight_layout()
    block(
        "vous", "6.1 · Stéréo : disparité et précision",
        "*Faites varier la distance : la disparité diminue, et l'erreur de profondeur **explose**.*",
        mo.hstack([
            mo.vstack([mo.hstack([st_b, st_f], justify="start", wrap=True), mo.hstack([st_z, st_dd], justify="start", wrap=True)], gap=0.5),
            _fig,
            mo.md(f"""
    À **Z = {st_z.value} cm**, b = {st_b.value} mm, f = {st_f.value} px :

    - disparité **d = b·f / Z = {fr(_d, 1)} px**
    - erreur **ΔZ ≈ Z² / (b·f) · Δd = {fr(_dZ * 1000, 2)} mm**

    **Doubler la distance multiplie l'erreur par 4.** Petite baseline : précis de près, pas de loin.
                """),
        ], justify="start", gap=1.5, align="center", wrap=True),
        "*b = 18 mm et f sont des valeurs d'exemple : à vérifier sur la fiche technique de la caméra.*",
    )
    return


@app.cell(hide_code=True)
def _(
    block,
    depth,
    depth_scale,
    depth_to_color,
    figure,
    fr,
    frame,
    has_depth,
    mo,
    np,
    to_rgb,
):
    if not has_depth:
        _out = mo.callout(mo.md("⚠️ **Cette image n'a pas de profondeur** (fichier `nom_depth.npy` absent). "
                                "Utilisez l'exemple ou une capture caméra pour cette section."), kind="warn")
    else:
        _valid = depth > 0
        _m = depth[_valid].astype(np.float64) * depth_scale
        _out = block(
            "vous", "6.2 · Le tableau `depth`",
            f"`depth` : tableau `{depth.shape}` de type `{depth.dtype}`, indexé **[v, u]** comme l'image couleur (il lui est *aligné* avec `rs.align`).",
            mo.hstack([
                figure(to_rgb(frame["color"]), "*Couleur*", 420),
                figure(to_rgb(depth_to_color(depth, depth_scale)), "*Profondeur : 🔵 proche (5 cm) → 🔴 loin (60 cm), ⬛ invalide (0)*", 420),
            ], justify="start", gap=1.5, wrap=True),
            f"**0 = pas de mesure** (zone sans texture, reflet, trop près ou trop loin) : **{fr(100 * (~_valid).mean(), 1)} %** des pixels. "
            f"Distances mesurées : {fr(_m.min() * 100, 0)} → {fr(float(np.median(_m)) * 100, 0)} (médiane) → {fr(_m.max() * 100, 0)} cm.",
        )
    _out
    return


@app.cell(hide_code=True)
def _(sn_block, sn_editor):
    sed_zpx = sn_editor("zpx")
    sn_block("zpx", sed_zpx)
    return (sed_zpx,)


@app.cell(hide_code=True)
def _(depth, depth_scale, has_depth, mo, pu, pv, sed_zpx, sn_eval, sn_view):
    if not has_depth:
        z_px = None
        zpx_ok = False
        _out = mo.callout(mo.md("⚠️ Pas de profondeur pour cette image."), kind="warn")
    else:
        _sev = sn_eval("zpx", sed_zpx.value, depth=depth, depth_scale=float(depth_scale), pu=pu, pv=pv)
        z_px = _sev["values"]["z_px"]
        zpx_ok = _sev["good"]
        _out = sn_view(_sev)
    _out
    return z_px, zpx_ok


@app.cell(hide_code=True)
def _(mo):
    dk = mo.ui.slider(1, 15, step=2, value=5, label="Voisinage du pixel (côté, px)", show_value=True)
    return (dk,)


@app.cell(hide_code=True)
def _(
    block,
    depth,
    depth_scale,
    dk,
    fr,
    has_depth,
    mo,
    np,
    pu,
    pv,
    z_px,
    zpx_ok,
):
    mo.stop(not has_depth)
    mo.stop(not zpx_ok, mo.callout(mo.md("🔒 *Les résultats s'afficheront dès que **6.2** sera validé.*"), kind="neutral"))
    _valid = depth > 0
    _med = float(np.median(depth[_valid].astype(np.float64) * depth_scale))
    _r = dk.value // 2
    _roi = depth[max(0, pv - _r):pv + _r + 1, max(0, pu - _r):pu + _r + 1]
    _vv = _roi[_roi > 0]
    _local = None if _vv.size == 0 else float(np.median(_vv)) * depth_scale
    _plaus = 0.05 <= _med <= 1.5
    mo.ui.tabs({
        "🎯 Au pixel choisi": block(
            "vous", "votre profondeur au pixel choisi",
            dk,
            mo.md(f"""
    | Pixel ({pu}, {pv}) | Valeur |
    |---|---|
    | brut `depth[{pv}, {pu}]` | **{int(depth[pv, pu])}** {'→ ⚠️ invalide (0)' if depth[pv, pu] == 0 else ''} |
    | **votre `z_px`** (mètres) | **{'—' if depth[pv, pu] == 0 else fr(z_px, 4) + ' m'}** |
    | médiane du voisinage {dk.value}×{dk.value} | {'—' if _local is None else fr(_local * 100, 1) + ' cm'} ({fr(100 * _vv.size / _roi.size, 0)} % valides) |
                """),
            "Un pixel isolé est **fragile** : s'il n'a pas de mesure il vaut 0, et son voisin peut différer de plusieurs millimètres. C'est ce qui motive 6.3.",
        ),
        "⚠️ Piège : depth_scale": block(
            "piege", "ne supposez pas l'unité",
            f"La profondeur brute est un entier ; `Z = brut × depth_scale`. Ici `depth_scale = {depth_scale}` (m par unité).",
            "Cette valeur est **propre à la caméra** : on la lit avec `sensor.get_depth_scale()`. Ne la devinez pas (1 mm ? 0,1 mm ?).",
            f"{'✅ La médiane mesurée (' + fr(_med * 100, 0) + ' cm) est plausible pour la D405 (7 à 50 cm).' if _plaus else '❌ Distance non plausible (' + fr(_med * 100, 0) + ' cm) : vérifiez depth_scale !'}",
        ),
    })
    return


@app.cell(hide_code=True)
def _(sn_block, sn_editor):
    sed_objmask = sn_editor("objmask")
    sn_block("objmask", sed_objmask)
    return (sed_objmask,)


@app.cell(hide_code=True)
def _(c_obj, figure, mask_clean, mo, sed_objmask, sn_eval, sn_view):
    if c_obj is None:
        obj_mask = None
        objmask_ok = False
        _out = mo.callout(mo.md("⚠️ **Aucun objet** (section 4) : élargissez les bornes de la section 3 ou baissez l'aire minimale."), kind="warn")
    else:
        _sev = sn_eval("objmask", sed_objmask.value, c_obj=c_obj, mask_clean=mask_clean)
        obj_mask = _sev["values"]["obj_mask"]
        objmask_ok = _sev["good"]
        _extra = figure(obj_mask, "*Votre masque de l'objet inspecté*", 380) if objmask_ok else None
        _out = sn_view(_sev, _extra)
    _out
    return obj_mask, objmask_ok


@app.cell(hide_code=True)
def _(sn_block, sn_editor):
    sed_zobj = sn_editor("zobj")
    sn_block("zobj", sed_zobj)
    return (sed_zobj,)


@app.cell(hide_code=True)
def _(depth, depth_scale, has_depth, mo, obj_mask, sed_zobj, sn_eval, sn_view):
    if not has_depth or obj_mask is None:
        z_obj = None
        zobj_ok = False
        _out = mo.callout(mo.md("⚠️ Il faut une **profondeur** et un **objet détecté** (section 4) pour cette étape."), kind="warn")
    else:
        _sev = sn_eval("zobj", sed_zobj.value, depth=depth, obj_mask=obj_mask, depth_scale=float(depth_scale))
        z_obj = _sev["values"]["z_obj"]
        zobj_ok = _sev["good"]
        _out = sn_view(_sev)
    _out
    return z_obj, zobj_ok


@app.cell(hide_code=True)
def _(
    block,
    depth,
    depth_scale,
    fr,
    has_depth,
    mo,
    obj_mask,
    objmask_ok,
    plt,
    u_i,
    v_i,
    z_obj,
    zobj_ok,
):
    mo.stop(not has_depth or obj_mask is None or z_obj is None)
    mo.stop(not (objmask_ok and zobj_ok), mo.callout(mo.md("🔒 *Les résultats s'afficheront dès que **6.3a** et **6.3b** seront validés.*"), kind="neutral"))
    _ok = (obj_mask > 0) & (depth > 0)                 # pixels du masque ET mesurés
    _vals = depth[_ok]
    _zc = int(depth[min(max(v_i, 0), depth.shape[0] - 1), min(max(u_i, 0), depth.shape[1] - 1)])   # au pixel du centroïde (fragile)
    _all = depth[obj_mask > 0]
    _mean0 = float(_all.mean()) * depth_scale          # moyenne naïve, zéros compris (faux)
    _fig, _ax = plt.subplots(figsize=(5.4, 2.6))
    _ax.hist(_vals * depth_scale * 100, bins=40, color="#5B7F95")
    _ax.axvline(z_obj * 100, color="#E8833A", lw=2.5, label=f"votre z_obj : {fr(z_obj * 100, 1)} cm")
    _ax.axvline(_mean0 * 100, color="#C0392B", lw=2, ls="--", label=f"moyenne avec les 0 : {fr(_mean0 * 100, 1)} cm")
    _ax.set_xlabel("profondeur des pixels de l'objet (cm)"); _ax.set_ylabel("pixels")
    _ax.legend(frameon=False, fontsize=9)
    for _s in ("top", "right"):
        _ax.spines[_s].set_visible(False)
    _fig.tight_layout()
    _zero = 100 * (1 - _vals.size / _all.size)
    mo.ui.tabs({
        "📏 Trois estimateurs": block(
            "vous", "6.3 · Votre profondeur, comparée à d'autres estimateurs",
            mo.hstack([
                _fig,
                mo.md(f"""
    | Estimateur | Valeur |
    |---|---|
    | au pixel du centroïde `depth[{v_i}, {u_i}]` | {_zc} → {'**invalide (0)**' if _zc == 0 else fr(_zc * depth_scale * 100, 1) + ' cm'} |
    | moyenne du masque (zéros compris) | {fr(_mean0 * 100, 1)} cm ❌ |
    | **votre `z_obj`** : pixels valides du masque | **{fr(z_obj * 100, 1)} cm** ✅ ({_vals.size} px sur {_all.size}) |
                    """),
            ], justify="start", gap=1.5, align="center", wrap=True),
        ),
        "⚠️ Piège : les zéros et les bords": block(
            "piege", "pourquoi pas le pixel du centroïde ?",
            f"Dans le masque de l'objet, **{fr(_zero, 1)} %** des pixels n'ont **pas de mesure** (0). Le pixel du centroïde peut en faire partie.",
            "Une **moyenne** qui inclut ces zéros tire la distance vers 0 : une valeur fausse mais qui a l'air plausible.",
            "Les pixels de **bord** mélangent objet et fond : la **médiane** les ignore, la moyenne non.",
        ),
    })
    return


@app.cell(hide_code=True)
def _(sn_block, sn_editor):
    sed_xyz = sn_editor("xyz")
    sn_block("xyz", sed_xyz)
    return (sed_xyz,)


@app.cell(hide_code=True)
def _(K, mo, np, sed_xyz, sn_eval, sn_view, u_c, v_c, z_obj):
    if z_obj is None or u_c is None:
        xyz_obj = None
        xyz_ok = False
        _out = mo.callout(mo.md("⚠️ Il faut un **centroïde** (section 5) **et** une **profondeur** (6.3) pour cette étape."), kind="warn")
    else:
        _sev = sn_eval("xyz", sed_xyz.value, u_c=float(u_c), v_c=float(v_c), z_obj=float(z_obj), K=K)
        _v = _sev["values"]
        xyz_obj = np.array([_v["X"], _v["Y"], _v["Z"]], dtype=float)
        xyz_ok = _sev["good"]
        _out = sn_view(_sev)
    _out
    return xyz_obj, xyz_ok


@app.cell(hide_code=True)
def _(K, block, cx, cy, fr, fx, fy, mo, np, u_c, v_c, xyz_obj, xyz_ok, z_obj):
    mo.stop(xyz_obj is None)
    mo.stop(not xyz_ok, mo.callout(mo.md("🔒 *Les résultats s'afficheront dès que **6.4** sera validé.*"), kind="neutral"))
    _P2 = z_obj * np.linalg.inv(K) @ np.array([u_c, v_c, 1.0])      # forme matricielle
    X, Y, Z = (float(a) for a in xyz_obj)
    block(
        "vous", "6.4 · Votre point 3D (rétro-projection)",
        mo.md(f"""
    $$X = \\frac{{Z}}{{f_x}}\\,(u - c_x) \\qquad Y = \\frac{{Z}}{{f_y}}\\,(v - c_y) \\qquad
    \\begin{{pmatrix}}X\\\\Y\\\\Z\\end{{pmatrix}} = Z\\,K^{{-1}}\\begin{{pmatrix}}u\\\\v\\\\1\\end{{pmatrix}}$$
            """),
        f"Avec (u<sub>c</sub>, v<sub>c</sub>) = ({fr(u_c, 1)} ; {fr(v_c, 1)}) px, Z = {fr(z_obj, 4)} m, "
        f"f<sub>x</sub> = {fr(fx, 1)}, f<sub>y</sub> = {fr(fy, 1)}, (c<sub>x</sub>, c<sub>y</sub>) = ({fr(cx, 1)} ; {fr(cy, 1)}) :",
        mo.md(f"### P = ( {fr(X * 100, 1)} cm ; {fr(Y * 100, 1)} cm ; {fr(Z * 100, 1)} cm )  — repère de la caméra"),
        f"Forme matricielle : ({fr(_P2[0] * 100, 1)} ; {fr(_P2[1] * 100, 1)} ; {fr(_P2[2] * 100, 1)}) cm → "
        f"{'✅ identique' if np.allclose(_P2, xyz_obj) else '❌ différent'} · distance à la caméra ‖P‖ = **{fr(np.linalg.norm(xyz_obj) * 100, 1)} cm**",
        mo.accordion({"📦 Avec le SDK de la caméra": mo.md(
            "`pyrealsense2` sait le faire : `rs.rs2_deproject_pixel_to_point(intrinsics, [u, v], z)` renvoie le même point, en lisant "
            "les intrinsèques de l'image alignée (celles de la couleur). Vous venez d'en écrire le contenu.")}),
        "Ensuite : repère **caméra → robot** par la transformation `[R | t]` (cours, slides 28-30).",
    )
    return


@app.cell(hide_code=True)
def _(ex_block, ex_editor):
    ed_od = ex_editor("object_depth")
    ex_block("object_depth", ed_od)
    return (ed_od,)


@app.cell(hide_code=True)
def _(c_obj, depth, depth_scale, ed_od, evaluate, ex_view, has_depth, z_obj):
    ev_od = evaluate("object_depth", ed_od.value)
    _extra = None
    if ev_od["fn"] is not None and ev_od["good"] and has_depth and c_obj is not None and z_obj is not None:
        try:
            _r = ev_od["fn"](depth, c_obj, float(depth_scale))
            _same = _r is not None and abs(_r - z_obj) < 1e-9
            _extra = "Sur **votre objet inspecté** : " + ("✅ même profondeur qu'en 6.3." if _same else "❌ résultat différent de celui de 6.3.")
        except Exception as _e:
            _extra = f"Sur votre objet : ❌ {type(_e).__name__} : {_e}"
    ex_view(ev_od, _extra)
    return (ev_od,)


@app.cell(hide_code=True)
def _(ex_block, ex_editor):
    ed_dp = ex_editor("deproject")
    ex_block("deproject", ed_dp)
    return (ed_dp,)


@app.cell(hide_code=True)
def _(K, ed_dp, evaluate, ex_view, np, u_c, v_c, xyz_obj, z_obj):
    ev_dp = evaluate("deproject", ed_dp.value)
    _extra = None
    if ev_dp["fn"] is not None and ev_dp["good"] and xyz_obj is not None:
        try:
            _r = ev_dp["fn"](u_c, v_c, z_obj, K)
            _same = np.allclose(np.asarray(_r, float).reshape(-1), xyz_obj, atol=1e-9)
            _extra = "Sur **votre objet** : " + ("✅ même point 3D qu'en 6.4." if _same else "❌ résultat différent de celui de 6.4.")
        except Exception as _e:
            _extra = f"Sur votre objet : ❌ {type(_e).__name__} : {_e}"
    ex_view(ev_dp, _extra)
    return (ev_dp,)


@app.cell(hide_code=True)
def _(block, fr, xyz_obj):
    if xyz_obj is None:
        _out = None
    else:
        _out = block(
            "retenir", "section 6",
            f"""
    - La profondeur brute est un **entier** : `Z = brut × depth_scale` ; **0 = invalide** ; `depth_scale` se **lit** sur la caméra. Indexer **`[v, u]`**.
    - Profondeur d'un objet : on fabrique son **masque** (intérieur du contour rempli), puis la **médiane** des pixels **du masque et valides** (`depth > 0`), pas le pixel du centroïde ni une moyenne avec les 0.
    - Rétro-projection : `X = Z (u − cx) / fx`, `Y = Z (v − cy) / fy` → ici P = ({fr(xyz_obj[0] * 100, 1)}, {fr(xyz_obj[1] * 100, 1)}, {fr(xyz_obj[2] * 100, 1)}) cm.
    - L'erreur de profondeur croît comme **Z²** : travaillez dans la plage 7 à 50 cm.
    - Pas de mesure valide ou pas d'objet : on renvoie `None`, on ne devine pas.
                """,
        )
    _out
    return


@app.cell(hide_code=True)
def _(block, mo):
    mo.vstack([
        mo.md("---\n# 7 · Marqueurs ArUco"),
        block("rappel", "un repère de position connue dans la scène",
              r"""
    Un **ArUco** est un carré noir à motif binaire **n×n** : détectable sans couleur, il donne un **identifiant** et ses **4 coins** (ordre fixe : haut-gauche → haut-droit → bas-droit → bas-gauche).
    Avec la taille réelle $L$ et les intrinsèques $K$, `solvePnP` retrouve sa **pose** `[R | t]` dans le repère caméra.

    Détection en 4 étapes : ① niveaux de gris + **seuil adaptatif local** → ② contours, quadrilatères convexes → ③ **rectification** (homographie) + Otsu + lecture des bits → ④ comparaison au **dictionnaire**.

    À **écrire** : **7.2** la détection · **7.3a** les coins 3D · **7.3b** la pose · **7.5** les fonctions. **7.1** et **7.4** sont des observations.
                  """),
    ])
    return


@app.cell(hide_code=True)
def _(cv2, mo):
    _dicts = {n: getattr(cv2.aruco, n) for n in
              ["DICT_4X4_50", "DICT_5X5_50", "DICT_6X6_50", "DICT_ARUCO_ORIGINAL"] if hasattr(cv2.aruco, n)}
    dict_dd = mo.ui.dropdown(options=list(_dicts), value="DICT_4X4_50", label="Dictionnaire")
    gen_id = mo.ui.number(0, 49, value=7, label="id à générer")
    marker_len = mo.ui.number(0.01, 0.30, step=0.005, value=0.07, label="Côté L du marqueur (m)")
    return dict_dd, gen_id, marker_len


@app.cell(hide_code=True)
def _(block, cv2, dict_dd, gen_id, marker_len, mo):
    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_dd.value))
    _n = aruco_dict.markerSize
    _gid = int(min(gen_id.value, aruco_dict.bytesList.shape[0] - 1))
    _mk = cv2.aruco.generateImageMarker(aruco_dict, _gid, 8 * (_n + 2))       # 8 px par case
    _bits = _mk[::8, ::8][1:-1, 1:-1] // 255                                  # la grille de bits (sans bordure)
    _png = cv2.imencode(".png", cv2.copyMakeBorder(_mk, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=255))[1].tobytes()
    _dl = mo.download(data=_png, filename=f"aruco_{dict_dd.value}_{_gid}.png", label="⬇️ PNG à imprimer")
    _grid = chr(10).join(" ".join(str(int(b)) for b in row) for row in _bits)
    block(
        "vous", "7.1 · Générer et imprimer un marqueur",
        mo.hstack([dict_dd, gen_id, marker_len], justify="start", gap=1.5, wrap=True),
        mo.hstack([
            mo.image(cv2.cvtColor(cv2.copyMakeBorder(_mk, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255), cv2.COLOR_GRAY2RGB), width=170),
            mo.md(f"""
    **Marqueur {_gid}** — {dict_dd.value} : bordure noire + grille **{_n}×{_n}** = {_n * _n} bits, {aruco_dict.bytesList.shape[0]} id possibles.

    ```
    {_grid}
    ```
    Les 4 rotations d'un même marqueur donnent 4 motifs **distincts** : l'orientation est donc connue.
                """),
            _dl,
        ], justify="start", gap=1.5, align="start", wrap=True),
        "Imprimez-le **avec une marge blanche** et mesurez son côté noir réel : c'est le **L** ci-dessus (en mètres), indispensable pour la pose.",
    )
    return (aruco_dict,)


@app.cell(hide_code=True)
def _(mo):
    ar_win_min = mo.ui.slider(3, 53, step=2, value=3, label="Fenêtre seuil adaptatif min (px)", show_value=True)
    ar_win_max = mo.ui.slider(3, 103, step=2, value=23, label="Fenêtre seuil adaptatif max (px)", show_value=True)
    ar_const = mo.ui.slider(0, 20, value=7, label="Constante C (seuil = moyenne locale − C)", show_value=True)
    ar_minper = mo.ui.slider(0.01, 0.2, step=0.01, value=0.03, label="Périmètre minimal (fraction de l'image)", show_value=True)
    return ar_const, ar_minper, ar_win_max, ar_win_min


@app.cell(hide_code=True)
def _(ar_const, ar_minper, ar_win_max, ar_win_min, cv2):
    # réglages du détecteur, construits à partir des curseurs de 7.2 (valeurs par défaut d'OpenCV au départ)
    params = cv2.aruco.DetectorParameters()
    params.adaptiveThreshWinSizeMin = int(ar_win_min.value)
    params.adaptiveThreshWinSizeMax = int(max(ar_win_max.value, ar_win_min.value))
    params.adaptiveThreshConstant = float(ar_const.value)
    params.minMarkerPerimeterRate = float(ar_minper.value)
    return (params,)


@app.cell(hide_code=True)
def _(sn_block, sn_editor):
    sed_aruco = sn_editor("aruco")
    sn_block("aruco", sed_aruco)
    return (sed_aruco,)


@app.cell(hide_code=True)
def _(aruco_dict, img, params, sed_aruco, sn_eval, sn_view):
    sev_aruco = sn_eval("aruco", sed_aruco.value, img=img, aruco_dict=aruco_dict, params=params)
    _v = sev_aruco["values"]
    gray, corners, ids, rejected = _v["gray"], _v["corners"], _v["ids"], _v["rejected"]
    aruco_ok = sev_aruco["good"]
    n_found = 0 if ids is None else len(ids)
    sn_view(sev_aruco)
    return aruco_ok, corners, gray, ids, n_found, rejected


@app.cell(hide_code=True)
def _(
    ar_const,
    ar_minper,
    ar_win_max,
    ar_win_min,
    aruco_ok,
    block,
    corners,
    cv2,
    draw_markers,
    figure,
    frame,
    gray,
    ids,
    mo,
    n_found,
    np,
    rejected,
    to_rgb,
):
    mo.stop(not aruco_ok, mo.callout(mo.md("🔒 *Les résultats s'afficheront dès que **7.2** sera validé.*"), kind="neutral"))
    # ---- les 4 étapes, illustrées (fenêtre médiane des réglages)
    _w = int(max(3, (ar_win_min.value + ar_win_max.value) // 2) | 1)
    _thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, _w, ar_const.value)
    _quads = frame["color"].copy()
    _cnt, _ = cv2.findContours(_thr, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    _nq = 0
    for _c in _cnt:
        if cv2.contourArea(_c) < 300:
            continue
        _ap = cv2.approxPolyDP(_c, 0.03 * cv2.arcLength(_c, True), True)
        if len(_ap) == 4 and cv2.isContourConvex(_ap):
            cv2.polylines(_quads, [_ap], True, (0, 200, 255), 2); _nq += 1
    _det = frame["color"].copy()
    draw_markers(_det, corners, ids)
    _rej = frame["color"].copy()
    cv2.polylines(_rej, [r.astype(np.int32) for r in rejected], True, (60, 60, 230), 2)

    # ---- rectification + bits du premier marqueur détecté
    if n_found:
        _src = corners[0].reshape(4, 2).astype(np.float32)
        _S = 70
        _dst = np.array([[0, 0], [_S - 1, 0], [_S - 1, _S - 1], [0, _S - 1]], np.float32)
        _warp = cv2.warpPerspective(gray, cv2.getPerspectiveTransform(_src, _dst), (_S, _S))
        _otsu, _bin = cv2.threshold(_warp, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        _rect_view = cv2.cvtColor(cv2.resize(_bin, (330, 330), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
        _rect_cap = f"*③ marqueur redressé (homographie) puis Otsu (seuil {int(_otsu)}) : on lit la grille de bits*"
    else:
        _rect_view = np.full((330, 330, 3), 200, np.uint8)
        _rect_cap = "*③ rien à redresser : aucun marqueur détecté*"

    mo.vstack([
        block(
            "vous", "7.2 · Votre détection, et ce qui se passe dedans",
            "*Réglez le détecteur : votre code de 7.2 est réévalué à chaque changement. Parcourez ensuite les 4 étapes dans les onglets.*",
            mo.hstack([ar_win_min, ar_win_max], justify="start", wrap=True),
            mo.hstack([ar_const, ar_minper], justify="start", wrap=True),
            f"**{n_found} marqueur(s) détecté(s)** · {len(rejected)} candidat(s) rejeté(s)",
            mo.hstack([figure(to_rgb(_det), f"*Résultat : {n_found} marqueur(s) identifié(s)*", 520)], justify="start"),
        ),
        mo.ui.tabs({
            "① Gris + seuil adaptatif": block(
                "rappel", "quel seuil ?",
                "On travaille en **niveaux de gris** (ni RGB ni HSV : le marqueur est noir et blanc) avec un seuil **adaptatif** : "
                "chaque pixel est comparé à la **moyenne de son voisinage moins C**, donc robuste aux ombres. Le seuil d'Otsu n'intervient qu'à l'étape ③.",
                figure(to_rgb(cv2.cvtColor(_thr, cv2.COLOR_GRAY2BGR)), f"*① seuil adaptatif (fenêtre {_w}, C = {ar_const.value})*", 520),
            ),
            "② Candidats carrés": block(
                "rappel", "contours puis polygones à 4 côtés",
                "On extrait les contours de l'image binaire, on les simplifie (`approxPolyDP`) et on ne garde que les **quadrilatères convexes** d'assez grand périmètre.",
                figure(to_rgb(_quads), f"*② {_nq} quadrilatères convexes candidats*", 520),
            ),
            "③ Redressement et bits": block(
                "rappel", "on lit la grille",
                "Une homographie ramène le candidat à un **carré**, on binarise (Otsu) et on lit le centre de chaque cellule : blanc = 1, noir = 0.",
                figure(to_rgb(_rect_view), _rect_cap, 330),
            ),
            "④ Dictionnaire": block(
                "rappel", "identifier",
                "La grille lue est comparée à tous les motifs du dictionnaire, dans les 4 rotations : le plus proche (tolérance de quelques bits) donne l'**id** et l'**ordre des coins**.",
                f"Votre résultat : `corners` = {len(corners)} tableau(x) · `ids` = {None if ids is None else ids.ravel().tolist()}",
            ),
            "⚠️ Pas détecté ?": block(
                "piege", "les causes habituelles",
                "- **Mauvais dictionnaire** : le marqueur imprimé et le dictionnaire du code doivent correspondre.\n"
                "- **Pas de marge blanche** autour du cadre noir : le carré n'est pas isolé.\n"
                "- **Trop petit ou flou** : baissez le périmètre minimal, rapprochez le marqueur.\n"
                "- **Reflets, angle rasant, ombre dure** sur le cadre.",
                f"Ici : {n_found} détecté(s), {len(rejected)} rejeté(s), périmètre minimal = {ar_minper.value}.",
            ),
        }),
    ])
    return


@app.cell(hide_code=True)
def _(sn_block, sn_editor):
    sed_obj3d = sn_editor("obj3d")
    sn_block("obj3d", sed_obj3d)
    return (sed_obj3d,)


@app.cell(hide_code=True)
def _(marker_len, np, sed_obj3d, sn_eval, sn_view):
    sev_obj3d = sn_eval("obj3d", sed_obj3d.value, marker_len=float(marker_len.value))
    obj = sev_obj3d["values"]["obj"]
    obj3d_ok = sev_obj3d["good"]
    _extra = None
    if obj3d_ok:
        _rows = ["| coin | x (cm) | y (cm) | z (cm) |", "|---|---|---|---|"]
        for _n, _p in zip(["haut-gauche", "haut-droit", "bas-droit", "bas-gauche"], np.asarray(obj, float)):
            _rows.append(f"| {_n} | {_p[0] * 100:+.2f} | {_p[1] * 100:+.2f} | {_p[2] * 100:+.2f} |")
        _extra = "Vos 4 coins, dans le repère du marqueur (origine au centre) :\n\n" + "\n".join(_rows)
    sn_view(sev_obj3d, _extra)
    return obj, obj3d_ok


@app.cell(hide_code=True)
def _(sn_block, sn_editor):
    sed_pnp = sn_editor("pnp")
    sn_block("pnp", sed_pnp)
    return (sed_pnp,)


@app.cell(hide_code=True)
def _(
    K,
    corners,
    dist_coeffs,
    ids,
    mo,
    n_found,
    np,
    obj,
    sed_pnp,
    sn_eval,
    sn_view,
):
    if not n_found:
        rvec = tvec = img_pts = marker_id = None
        pnp_ok = False
        _out = mo.callout(mo.md("⚠️ **Aucun marqueur détecté** : revenez à 7.2 (voir l'onglet « Pas détecté ? »)."), kind="warn")
    else:
        img_pts = corners[0].reshape(4, 2).astype(np.float32)          # coins du premier marqueur détecté
        marker_id = int(ids.ravel()[0])
        _sev = sn_eval("pnp", sed_pnp.value, obj=obj, img_pts=img_pts, K=K, dist_coeffs=dist_coeffs)
        _v = _sev["values"]
        rvec = None if _v["rvec"] is None else np.asarray(_v["rvec"], np.float64).reshape(3, 1)
        tvec = None if _v["tvec"] is None else np.asarray(_v["tvec"], np.float64).reshape(3, 1)
        pnp_ok = _sev["good"]
        _out = sn_view(_sev)
    _out
    return img_pts, marker_id, pnp_ok, rvec, tvec


@app.cell(hide_code=True)
def _(
    K,
    block,
    corners,
    cv2,
    dist_coeffs,
    draw_markers,
    fr,
    frame,
    ids,
    img_pts,
    marker_id,
    marker_len,
    mo,
    np,
    obj,
    obj3d_ok,
    pnp_ok,
    rvec,
    to_rgb,
    tvec,
):
    mo.stop(tvec is None)
    mo.stop(not (obj3d_ok and pnp_ok), mo.callout(mo.md("🔒 *Les résultats s'afficheront dès que **7.3a** et **7.3b** seront validés.*"), kind="neutral"))
    _L = float(marker_len.value)
    _R, _ = cv2.Rodrigues(rvec)
    _pr, _ = cv2.projectPoints(np.asarray(obj, np.float32), rvec, tvec, K, dist_coeffs)
    _reproj = float(np.linalg.norm(_pr.reshape(4, 2) - img_pts, axis=1).mean())
    _view = frame["color"].copy()
    draw_markers(_view, corners, ids)
    cv2.drawFrameAxes(_view, K, dist_coeffs, rvec, tvec, _L * 0.8, 3)
    _t = tvec.ravel()
    block(
        "vous", f"7.3 · La pose du marqueur {marker_id}",
        mo.hstack([
            mo.image(to_rgb(_view), width=520),
            mo.md(f"""
    Coins (u, v) : {', '.join(f'({p[0]:.1f}, {p[1]:.1f})' for p in img_pts)}

    **t** = ({fr(_t[0] * 100, 1)} ; {fr(_t[1] * 100, 1)} ; {fr(_t[2] * 100, 1)}) cm — position du centre du marqueur (repère caméra)

    distance ‖t‖ = **{fr(np.linalg.norm(_t) * 100, 1)} cm**

    erreur de reprojection moyenne = **{fr(_reproj, 2)} px** *(≈ 0 : le modèle colle aux coins détectés)*

    Axes dessinés : 🔴 x · 🟢 y · 🔵 z (sort du marqueur, vers la caméra).
                """),
        ], justify="start", gap=1.5, align="center", wrap=True),
        mo.accordion({"🔄 La matrice de rotation R (Rodrigues)": mo.md(f"```\n{np.array2string(_R, precision=3, suppress_small=True)}\n```")}),
    )
    return


@app.cell(hide_code=True)
def _(
    block,
    corners,
    cv2,
    depth,
    depth_scale,
    fr,
    has_depth,
    mo,
    n_found,
    np,
    pnp_ok,
    tvec,
):
    mo.stop(not n_found or tvec is None)
    mo.stop(not pnp_ok, mo.callout(mo.md("🔒 *Les résultats s'afficheront dès que **7.3b** sera validé.*"), kind="neutral"))
    _z_pnp = float(tvec.ravel()[2])
    if not has_depth:
        _out = block("vous", "7.4 · Deux mesures de la distance", f"Z(solvePnP) = **{fr(_z_pnp * 100, 1)} cm** (pas de profondeur à comparer).")
    else:
        _m = np.zeros(depth.shape, np.uint8)
        cv2.fillConvexPoly(_m, corners[0].reshape(4, 2).astype(np.int32), 255)
        _v = depth[(_m > 0) & (depth > 0)]
        if _v.size == 0:
            _out = block("vous", "7.4 · Deux mesures de la distance", "Aucune profondeur valide dans le marqueur.")
        else:
            _zd = float(np.median(_v)) * depth_scale
            _e = (_zd - _z_pnp) * 1000
            _out = block(
                "vous", "7.4 · Deux mesures de la distance au marqueur",
                mo.md(f"""
    | Source | Z |
    |---|---|
    | `solvePnP` (géométrie : taille connue **L** + intrinsèques) | **{fr(_z_pnp * 100, 1)} cm** |
    | caméra de profondeur (médiane dans le quadrilatère, {_v.size} px) | **{fr(_zd * 100, 1)} cm** |
    | écart | {fr(_e, 1)} mm |
                    """),
                "Deux principes **indépendants** qui s'accordent : bon signe. Un grand écart indique un **L faux**, des intrinsèques erronées ou un `depth_scale` incorrect.",
            )
    _out
    return


@app.cell(hide_code=True)
def _(ex_block, ex_editor):
    ed_da = ex_editor("detect_aruco")
    ex_block("detect_aruco", ed_da)
    return (ed_da,)


@app.cell(hide_code=True)
def _(cv2, dict_dd, ed_da, evaluate, ex_view, ids, img):
    ev_da = evaluate("detect_aruco", ed_da.value)
    _extra = None
    if ev_da["fn"] is not None and ev_da["good"]:
        try:
            _r = ev_da["fn"](img, getattr(cv2.aruco, dict_dd.value))
            _mine = sorted(int(i) for i, _ in _r)
            _ref = [] if ids is None else sorted(int(i) for i in ids.ravel())
            _extra = "Sur **votre image** : " + (f"✅ mêmes marqueurs qu'en 7.2 ({_ref})." if _mine == _ref else f"❌ ids {_mine} au lieu de {_ref} (7.2).")
        except Exception as _e:
            _extra = f"Sur votre image : ❌ {type(_e).__name__} : {_e}"
    ex_view(ev_da, _extra)
    return (ev_da,)


@app.cell(hide_code=True)
def _(ex_block, ex_editor):
    ed_mp = ex_editor("marker_pose")
    ex_block("marker_pose", ed_mp)
    return (ed_mp,)


@app.cell(hide_code=True)
def _(K, dist_coeffs, ed_mp, evaluate, ex_view, img_pts, marker_len, np, tvec):
    ev_mp = evaluate("marker_pose", ed_mp.value)
    _extra = None
    if ev_mp["fn"] is not None and ev_mp["good"] and img_pts is not None and tvec is not None:
        try:
            _r = ev_mp["fn"](img_pts, float(marker_len.value), K, dist_coeffs)
            _same = _r is not None and np.linalg.norm(np.asarray(_r[1], float).reshape(-1) - tvec.reshape(-1)) < 5e-3
            _extra = "Sur **votre marqueur** : " + ("✅ même position qu'en 7.3." if _same else "❌ position différente de celle de 7.3.")
        except Exception as _e:
            _extra = f"Sur votre marqueur : ❌ {type(_e).__name__} : {_e}"
    ex_view(ev_mp, _extra)
    return (ev_mp,)


@app.cell(hide_code=True)
def _(block, fr, marker_id, tvec):
    if tvec is None:
        _out = None
    else:
        _t = tvec.ravel()
        _out = block(
            "retenir", "section 7",
            f"""
    - La détection travaille sur l'image en **niveaux de gris**, avec **le même dictionnaire** que celui imprimé ; elle renvoie `corners`, `ids`, `rejected`.
    - **`ids` vaut `None` quand rien n'est détecté** : toujours le tester avant de l'utiliser.
    - La pose exige la **taille réelle L** (coins 3D dans le repère du marqueur, centre en origine) et les intrinsèques **K** ; les coins 3D et 2D doivent être dans **le même ordre**.
    - Ici id {marker_id}, t = ({fr(_t[0] * 100, 1)}, {fr(_t[1] * 100, 1)}, {fr(_t[2] * 100, 1)}) cm.
    - Le marqueur est un **repère de référence** dans la scène : on le retrouvera pour passer du repère caméra au repère robot.
                """,
        )
    _out
    return


@app.cell(hide_code=True)
def _(block, mo):
    mo.vstack([
        mo.md("---\n# 8 · Tout assembler : `process_frame`"),
        block("rappel", "d'une image à la liste des objets 3D",
              "Vous savez maintenant détecter, localiser et mesurer **un** objet. Le nœud de perception doit le faire pour **tous les objets, à chaque image** : "
              "une seule fonction, `process_frame`, enchaîne toutes vos briques. C'est elle que `vision_node.py` appellera."),
    ])
    return


@app.cell(hide_code=True)
def _(ex_block, ex_editor):
    ed_pf = ex_editor("process_frame")
    ex_block("process_frame", ed_pf)
    return (ed_pf,)


@app.cell(hide_code=True)
def _(
    K,
    depth,
    depth_scale,
    ed_pf,
    ev_cen,
    ev_clean,
    ev_dp,
    ev_fo,
    ev_hsv,
    ev_od,
    evaluate,
    ex_view,
    has_depth,
    hi,
    img,
    k_close,
    k_open,
    lo,
    min_area,
):
    _brique = {"hsv_mask": ev_hsv["used"], "clean_mask": ev_clean["used"], "find_objects": ev_fo["used"],
               "centroid": ev_cen["used"], "object_depth": ev_od["used"], "deproject": ev_dp["used"]}
    ev_pf = evaluate("process_frame", ed_pf.value, _brique)
    hsv_range = {"h_min": int(lo[0]), "s_min": int(lo[1]), "v_min": int(lo[2]),
                 "h_max": int(hi[0]), "s_max": int(hi[1]), "v_max": int(hi[2])}
    pf_objects = None
    _extra = None
    if ev_pf["fn"] is not None and ev_pf["good"]:
        if not has_depth:
            _extra = "Votre image n'a pas de profondeur : l'essai sur votre image est impossible."
        else:
            try:
                pf_objects = list(ev_pf["fn"](img, depth, K, float(depth_scale), hsv_range, int(min_area.value), int(k_open.value), int(k_close.value)))
                _extra = f"Sur **votre image** (vos réglages des sections 3 et 4) : **{len(pf_objects)} objet(s)** détecté(s)."
            except Exception as _e:
                _extra = f"Sur votre image : ❌ {type(_e).__name__} : {_e}"
    ex_view(ev_pf, _extra)
    return ev_pf, pf_objects


@app.cell(hide_code=True)
def _(block, cv2, ev_pf, fr, img, mo, np, pf_objects, to_rgb):
    mo.stop(pf_objects is None and ev_pf["good"])
    mo.stop(not ev_pf["good"], mo.callout(mo.md("🔒 *Le résultat sur votre image s'affichera dès que **`process_frame`** sera validée.*"), kind="neutral"))
    _view = img.copy()
    _rows = ["| # | centre (u, v) | aire (px²) | profondeur | position (X ; Y ; Z) en cm |", "|:-:|:--|--:|--:|:--|"]
    for _i, _o in enumerate(pf_objects, 1):
        _u, _v = (int(round(t)) for t in _o["centroid"])
        cv2.drawContours(_view, [_o["contour"]], -1, (0, 255, 0), 2)
        cv2.circle(_view, (_u, _v), 5, (0, 255, 0), -1)
        cv2.putText(_view, str(_i), (_u + 8, _v - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(_view, str(_i), (_u + 8, _v - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        _X, _Y, _Z = (float(t) * 100 for t in _o["xyz_m"])
        _rows.append(f"| {_i} | ({_u}, {_v}) | {fr(_o['area'], 0)} | {fr(_o['depth_m'] * 100, 1)} cm | ({fr(_X, 1)} ; {fr(_Y, 1)} ; {fr(_Z, 1)}) |")
    block(
        "vous", "8 · Le résultat sur votre image",
        mo.hstack([mo.image(to_rgb(_view), width=520), mo.md("\n".join(_rows) if pf_objects else "*Aucun objet détecté avec ces réglages.*")],
                  justify="start", gap=1.5, align="center", wrap=True),
    )
    return


@app.cell(hide_code=True)
def _(block, ev_pf):
    block(
        "retenir", "section 8",
        """
- `process_frame` = **la chaîne complète** : `hsv_mask` → `clean_mask` → `find_objects`, puis, **pour chaque objet**, `centroid` → `object_depth` → `deproject`.
- Elle renvoie **une liste** (vide s'il n'y a rien) : l'aval (le robot) doit pouvoir traiter 0, 1 ou plusieurs objets.
- Un objet sans centroïde ou sans profondeur valide est **ignoré** plutôt qu'inventé.
- Le nœud `live/vision_node.py` appelle cette fonction à chaque image et publie le résultat sur ZeroMQ.
        """,
    ) if ev_pf["good"] else None
    return


@app.cell(hide_code=True)
def _(mo):
    get_exp, set_exp = mo.state("")
    return get_exp, set_exp


@app.cell(hide_code=True)
def _(
    block,
    ev_cen,
    ev_clean,
    ev_da,
    ev_dp,
    ev_fo,
    ev_hsv,
    ev_mp,
    ev_od,
    ev_pf,
    get_exp,
    mo,
):
    export_btn = mo.ui.run_button(label="💾 Exporter dans ma_vision.py", kind="success")
    _evs = [ev_hsv, ev_clean, ev_fo, ev_cen, ev_od, ev_dp, ev_pf, ev_da, ev_mp]
    _ok = [e["name"] for e in _evs if e["good"]]
    _ko = [e["name"] for e in _evs if not e["good"]]
    mo.vstack([
        mo.md("---\n# 💾 Exporter votre module"),
        block(
            "coder", "Exporter votre module",
            "Les fonctions **validées** sont écrites dans `ma_vision.py`, dans le dossier du TP (un niveau au-dessus de ce notebook) ; `live/vision_node.py` les utilisera.",
            mo.md("- Prêtes à exporter : " + (", ".join(f"`{n}`" for n in _ok) or "aucune") + "\n"
                  + "- Pas encore validées (non exportées) : " + (", ".join(f"`{n}`" for n in _ko) or "aucune")),
            export_btn,
            mo.md(get_exp()) if get_exp() else mo.md(""),
        ),
    ])
    return (export_btn,)


@app.cell(hide_code=True)
def _(
    MA_VISION,
    ev_cen,
    ev_clean,
    ev_da,
    ev_dp,
    ev_fo,
    ev_hsv,
    ev_mp,
    ev_od,
    ev_pf,
    exercices,
    export_btn,
    mo,
    set_exp,
):
    mo.stop(not export_btn.value)
    try:
        with open(MA_VISION, encoding="utf-8") as _f:
            _existing = _f.read()
    except OSError:
        _existing = None
    _codes = {e["name"]: e["code"] for e in (ev_hsv, ev_clean, ev_fo, ev_cen, ev_od, ev_dp, ev_pf, ev_da, ev_mp) if e["good"]}
    with open(MA_VISION, "w", encoding="utf-8") as _f:
        _f.write(exercices.build_module(_codes, _existing))
    set_exp("✅ **`ma_vision.py` mis à jour** : " + (", ".join(f"`{n}`" for n in _codes) or "rien à exporter")
            + ". Vérification dans un terminal (dossier du TP) : `python notebook/checks.py`")
    return


if __name__ == "__main__":
    app.run()
