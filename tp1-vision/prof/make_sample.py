"""
make_sample.py — génère l'exemple fourni (captures/exemple.*)
=============================================================
Ce script (prof/) n'est PAS à distribuer aux étudiants ; il lit prof/_scene_src.png (photo de départ) : il fabrique une capture
synthétique mais *cohérente* (couleur + profondeur + intrinsèques) :
  - la couleur : photo du triton + un marqueur ArUco (id 7, DICT_4X4_50, côté 5 cm)
    incrusté avec une vraie perspective ;
  - la profondeur : plan du marqueur (le « bois »), l'animal est surélevé ;
    on ajoute du bruit, des trous (reflets) et des pixels de bord aberrants ;
  - les intrinsèques : f = 447 px, (cx, cy) = (424, 240), depth_scale = 0.0001.
Les valeurs de vérité (pose du marqueur, etc.) sont affichées en fin d'exécution.
"""
import json, os, cv2, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "eleves", "captures"); os.makedirs(OUT, exist_ok=True)   # écrit directement dans eleves/captures/
color = cv2.imread(os.path.join(HERE, "_scene_src.png"))
H, W = color.shape[:2]
F, CX, CY, SCALE = 447.0, 424.0, 240.0, 0.0001
K = np.array([[F, 0, CX], [0, F, CY], [0, 0, 1]])
def rot(ax, ay):
    rx = np.array([[1,0,0],[0,np.cos(ax),-np.sin(ax)],[0,np.sin(ax),np.cos(ax)]])
    ry = np.array([[np.cos(ay),0,np.sin(ay)],[0,1,0],[-np.sin(ay),0,np.cos(ay)]])
    return ry @ rx
R = rot(np.deg2rad(-25), np.deg2rad(28)) @ np.diag([1., -1., -1.])
t = np.array([-0.13, 0.075, 0.26])
n = R[:, 2]                                   # normale du plan du marqueur (repère caméra)
uu, vv = np.meshgrid(np.arange(W), np.arange(H))
ray = np.stack([(uu - CX) / F, (vv - CY) / F, np.ones_like(uu, float)], -1)
Zmk = (n @ t) / (ray @ n)                     # plan du marqueur (la feuille)
Zplane = 0.26 + 0.00035 * (371 - vv) + 0.0001 * (uu - 201)   # « bois » : plan doucement incliné
Lm = 0.05 * 1.4                                # feuille = marqueur + marge blanche
paper3d = np.array([[-Lm/2, Lm/2, 0], [Lm/2, Lm/2, 0], [Lm/2, -Lm/2, 0], [-Lm/2, -Lm/2, 0]])
pp, _ = cv2.projectPoints(paper3d, cv2.Rodrigues(R)[0], t.reshape(3, 1), K, np.zeros(5))
paper = np.zeros((H, W), np.uint8); cv2.fillConvexPoly(paper, pp.reshape(-1, 2).astype(np.int32), 255)
# masque de l'animal (même seuillage que dans les slides)
hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
m = cv2.inRange(hsv, (5, 100, 100), (20, 255, 255))
m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
obj = np.zeros_like(m); cv2.drawContours(obj, [max(cs, key=cv2.contourArea)], -1, 255, cv2.FILLED)
dist = cv2.distanceTransform(obj, cv2.DIST_L2, 5)
height = 0.012 + 0.040 * np.sqrt(np.clip(dist / dist.max(), 0, 1))   # 1,2 à 5,2 cm
Z = np.where(obj > 0, Zplane - height, np.where(paper > 0, Zmk, Zplane))
rng = np.random.default_rng(3)
Z = Z + rng.normal(0, 0.0004, Z.shape)                                   # bruit ~0,4 mm
# pixels de bord (mélange objet / fond) : 40 % prennent la profondeur du fond
edge = cv2.dilate(obj, np.ones((5, 5), np.uint8)) - cv2.erode(obj, np.ones((5, 5), np.uint8))
Z = np.where((edge > 0) & (rng.random(Z.shape) < 0.4), Zplane, Z)
# trous : reflets (S faible, V fort) + 0,3 % aléatoire
spec = (hsv[..., 1] < 60) & (hsv[..., 2] > 225)
spec = cv2.dilate(spec.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
invalid = spec | (rng.random(Z.shape) < 0.003) | (Z < 0.07) | (Z > 0.6)
raw = np.round(Z / SCALE).astype(np.int64); raw[invalid] = 0
raw = np.clip(raw, 0, 65535).astype(np.uint16)
cv2.imwrite(os.path.join(OUT, "exemple.png"), color)
np.save(os.path.join(OUT, "exemple_depth.npy"), raw)
json.dump({"width": W, "height": H, "fx": F, "fy": F, "ppx": CX, "ppy": CY,
           "dist": [0, 0, 0, 0, 0], "depth_scale": SCALE,
           "source": "exemple synthétique (photo + marqueur incrusté + profondeur calculée)"},
          open(os.path.join(OUT, "exemple.json"), "w"), indent=2, ensure_ascii=False)
v = raw[(obj > 0) & (raw > 0)]
print("profondeur de l'animal : médiane %.4f m, min %.3f, max %.3f" % (np.median(v) * SCALE, v.min() * SCALE, v.max() * SCALE))
print("Zmk au centre du marqueur : %.4f m (attendu %.3f)" % (Zmk[371, 201], t[2]))
print("pixels invalides : %.2f %%" % (100 * (raw == 0).mean()), " plage bois : %.3f–%.3f m" % (Zplane.min(), Zplane.max()))
print("VÉRITÉ marqueur : t =", t, " L = 0.05 m")
