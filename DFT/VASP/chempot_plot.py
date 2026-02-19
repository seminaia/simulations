from curses import use_default_colors
from operator import index
from tkinter import font
import adjustText
import pandas as pd
import numpy as np
from pymatgen.core import Composition
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, art3d
from scipy.spatial import ConvexHull
from scipy.optimize import linprog
from itertools import combinations
import re
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from sympy import plot
from adjustText import adjust_text
import matplotlib.cm as cm
import mplcursors
from math import gcd
import os

    # ---------- helpers ----------
def _is_floor_label(lbl: str) -> bool:
    return isinstance(lbl, str) and lbl.startswith("__FLOOR__")
def _color_for_label(lbl: str):
    # deterministic pastel-ish mapping without relying on index
    base = abs(hash(str(lbl))) % 997
    rng = np.random.RandomState(base)
    c = rng.rand(3) * 0.6 + 0.2
    return (float(c[0]), float(c[1]), float(c[2]))
def _order_polygon_3d(points: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Order coplanar points into a CCW loop by projecting onto plane (n,u,v)."""
    P = np.asarray(points, dtype=np.float64)
    c = P.mean(axis=0)
    n = np.asarray(normal, dtype=np.float64)
    n /= (np.linalg.norm(n) + 1e-15)
    # pick a reference not parallel to n
    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(ref, n)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    u = np.cross(n, ref); u /= (np.linalg.norm(u) + 1e-15)
    v = np.cross(n, u); v /= (np.linalg.norm(v) + 1e-15)
    UV = np.c_[(P - c) @ u, (P - c) @ v]
    ang = np.arctan2(UV[:, 1], UV[:, 0])
    return P[np.argsort(ang)]
def _parse_comp(formula: str):
    comp = Composition(str(formula).strip())
    comp = {el.symbol: float(comp[el]) for el in comp.elements}
    return comp
def _to_fu(E_val, n_tot, per_atom: bool) -> float:
    return float(E_val) * float(n_tot) if per_atom else float(E_val)
def _dedup_by_reduced(rows):
    """
    rows: list of (label, n1,n2,n3, Hfu).
    Keep lowest Hfu per *reduced* composition (normalized to integers).
    """
    best = {}
    for lab, n1, n2, n3, H in rows:
        vec = np.array([n1, n2, n3], dtype=np.float64)
        if vec.sum() <= 0:
            continue
        # reduced formula key via integer normalization
        g = np.gcd.reduce(np.round(vec).astype(int)) if np.allclose(vec, np.round(vec), atol=1e-8) else 1
        key = tuple(np.round(vec / (g if g else 1), 8))
        if (key not in best) or (H < best[key][1] - 1e-12):
            best[key] = (lab, float(H), vec)
    out = []
    for (lab, H, vec) in best.values():
        out.append((lab, float(vec[0]), float(vec[1]), float(vec[2]), float(H)))
    return out
   
# --- label decluttering helper ---
def _declutter_labels(ax, edge_texts, curve_texts):
    """
    Try to spread labels nicely. If 'adjustText' is available, use it.
    Otherwise, keep edge labels and hide overlapping curve labels.
    """
    # 1) Try adjustText (pip install adjustText)
    try:
        texts = edge_texts + curve_texts
        from adjustText import adjust_text
        # A few gentle nudges; keeps labels inside axes and apart
        adjust_text(
            texts, ax=ax,
            only_move={'text': 'xy'},
            expand_text=(1.1, 1.2),
            force_text=(0.08, 0.12),
            lim=300
        )
        return
    except Exception:
        pass
    # 2) Fallback: keep edges, drop overlapping curve labels
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    kept = []
 # keep all edge labels
    for t in edge_texts:
        bb = t.get_window_extent(renderer=renderer).expanded(1.05, 1.15)
        kept.append(bb)
    # add curve labels only if they don't overlap kept ones
    for t in curve_texts:
        bb = t.get_window_extent(renderer=renderer).expanded(1.05, 1.15)
        if any(bb.overlaps(b) for b in kept):
            t.set_visible(False)
        else:
            kept.append(bb)
def _parse(f):
    return {el:(float(n) if n else 1.0)
            for el,n in re.findall(r"([A-Z][a-z]?)(\d*\.?\d*)", str(f).strip())}
def plot_ternary_polyhedron(
    csv_path,
    elements, # e.g. ["La","Ni","O"]
    mu_floor=-20.0, # default floor for all axes
    mu_floor_by_el=None, # e.g. {"La":-6, "Ni":-3, "O":-2}
    out_png=None,
    label_faces=True,
    show_floor_faces=False, # draw μ floors as faint faces?
    tol=1e-8,
    face_alpha=0.35,
    edge_width=0.8,
    elev=24,
    azim=42,
    return_data=False, # return dict with vertices/faces/labels
    export_csv=None # e.g. "polyhedron_vertices_faces.csv"
):
    """
    Plot the 3D chemical-potential feasible region (A μ ≤ b) in Δμ_{E1,E2,E3} space.
    - Reads a CSV with 'Formula' and either 'Formation Energy (eV/atom)' or '(eV/fu)'.
    - Adds elemental cap planes μ_i ≤ 0 (even if not present in CSV).
    - Adds per-axis floors via mu_floor_by_el={"La":-6, "Ni":-3, "O":-2} (default uses mu_floor).
    - Enumerates vertices by intersecting all triples of planes and filtering by feasibility.
    - Draws each plane's tight polygonal face (unless it's a floor and show_floor_faces=False).
    Returns (if return_data=True): dict with keys:
      'vertices' (N×3), 'faces' (list of index lists), 'face_labels' (list), 'A','b','labels'
    """
    # ---------- inputs & CSV ----------
    if len(elements) != 3:
        raise ValueError("Need exactly three elements.")
    E1, E2, E3 = elements
    df = pd.read_csv(csv_path)
    col_per_atom = None
    if 'Formation Energy (eV/atom)' in df.columns:
        col_per_atom = 'Formation Energy (eV/atom)'; per_atom = True
    elif 'Formation Energy (eV/fu)' in df.columns:
        col_per_atom = 'Formation Energy (eV/fu)'; per_atom = False
    else:
        raise ValueError("CSV must contain 'Formation Energy (eV/atom)' or 'Formation Energy (eV/fu)'.")
    # per-axis floors
    floors = {E1: float(mu_floor), E2: float(mu_floor), E3: float(mu_floor)}
    if mu_floor_by_el:
        floors.update({k: float(v) for k, v in mu_floor_by_el.items() if k in floors})
    # ---------- collect phase planes ----------
    rows = [] # (label, n1,n2,n3, Hfu)
    seen_pure = {E1: False, E2: False, E3: False}
    for f, E in zip(df['Formula'].astype(str), df[col_per_atom]):
        try:
            c = _parse_comp(f)
        except Exception:
            continue
        if not set(c).issubset(set(elements)):
            continue
        n1, n2, n3 = float(c.get(E1, 0.0)), float(c.get(E2, 0.0)), float(c.get(E3, 0.0))
        N = n1 + n2 + n3
        if N <= 0:
            continue
        Hfu = _to_fu(E, N, per_atom)
        # Pure elemental caps: μ_i ≤ 0 (b=0), with clean labels
        if (n1 > 0) and (n2 == 0) and (n3 == 0):
            seen_pure[E1] = True; rows.append((E1, 1.0, 0.0, 0.0, 0.0)); continue
        if (n2 > 0) and (n1 == 0) and (n3 == 0):
            seen_pure[E2] = True; rows.append((E2, 0.0, 1.0, 0.0, 0.0)); continue
        if (n3 > 0) and (n1 == 0) and (n2 == 0):
            seen_pure[E3] = True; rows.append((E3, 0.0, 0.0, 1.0, 0.0)); continue
        rows.append((f, n1, n2, n3, float(Hfu)))
    # Deduplicate polymorphs by reduced composition (lowest ΔHf per fu)
    rows = _dedup_by_reduced(rows)
    # Add elemental caps if missing
    if not seen_pure[E1]: rows.append((E1, 1.0, 0.0, 0.0, 0.0))
    if not seen_pure[E2]: rows.append((E2, 0.0, 1.0, 0.0, 0.0))
    if not seen_pure[E3]: rows.append((E3, 0.0, 0.0, 1.0, 0.0))
    # Build A μ ≤ b with labels
    Arows, brows, labels = [], [], []
    for lab, n1, n2, n3, H in rows:
        Arows.append([n1, n2, n3]); brows.append(H); labels.append(lab)
    # Floors: -μ_i ≤ -floor_i
    Arows += [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]
    brows += [-floors[E1], -floors[E2], -floors[E3]]
    labels += [("__FLOOR__" + E1), ("__FLOOR__" + E2), ("__FLOOR__" + E3)]
    A = np.array(Arows, dtype=np.float64)
    b = np.array(brows, dtype=np.float64)
    m = A.shape[0]
    # ---------- enumerate feasible vertices ----------
    verts = []
    ids_triplets = list(combinations(range(m), 3))
    for ids in ids_triplets:
        M = A[list(ids), :]
        det = np.linalg.det(M)
        if abs(det) < 1e-12:
            continue
        mu = np.linalg.solve(M, b[list(ids)])
        if np.all(A @ mu <= b + 1e-8):
            verts.append(mu)
    if not verts:
        raise RuntimeError(
            "No feasible vertices found. "
            "Try loosening floors, check energy units, or verify formulas."
        )
    V = np.unique(np.round(np.array(verts, dtype=np.float64), 8), axis=0)
    if len(V) < 4:
        raise RuntimeError(
            f"Feasible region degenerate (|V|={len(V)}). "
            "Loosen μ floors or verify ΔHf values."
        )
    # ---------- draw faces per plane ----------
    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(111, projection='3d')
    faces = [] # list of index arrays into V
    face_labels = [] # label per face
    for k, (ak, bk, lab) in enumerate(zip(A, b, labels)):
        if _is_floor_label(lab) and not show_floor_faces:
            continue
        tight = np.abs(V @ ak - bk) <= max(tol, 1e-7)
        pts = V[tight]
        if len(pts) < 3:
            continue
        poly = _order_polygon_3d(pts, ak)
        facecolor = (0.5, 0.5, 0.5, 0.15) if _is_floor_label(lab) else _color_for_label(lab)
        alpha = 0.12 if _is_floor_label(lab) else face_alpha
        mesh = Poly3DCollection([poly], facecolors=[facecolor], alpha=alpha,
                                edgecolors='k', linewidths=edge_width)
        ax.add_collection3d(mesh)
        # record face as indices into V (preserve polygon order)
        idx = []
        for p in poly:
            # match rounded coordinates
            where = np.where(np.all(np.isclose(V, p[None, :], atol=1e-8), axis=1))[0]
            idx.append(int(where[0]))
        faces.append(idx)
        face_labels.append(lab)
        if label_faces and not _is_floor_label(lab):
            ctr = poly.mean(axis=0)
            n = ak / (np.linalg.norm(ak) + 1e-15)
            pos = ctr - 0.03 * n # nudge inside (A·μ ≤ b)
            if np.all(np.isfinite(pos)):
                ax.text(pos[0], pos[1], pos[2], str(lab),
                        fontsize=8, ha='center', va='center', zorder=10)
    # ---------- limits & cosmetics ----------
    mins = np.minimum(V.min(axis=0), np.array([floors[E1], floors[E2], floors[E3]], dtype=np.float64))
    maxs = np.maximum(V.max(axis=0), np.array([0.0, 0.0, 0.0], dtype=np.float64))
    for setter, lo, hi in zip([ax.set_xlim, ax.set_ylim, ax.set_zlim], mins, maxs):
        pad = 0.06 * max(1.0, (hi - lo))
        setter(lo - pad, hi + pad)
    ax.set_xlabel(rf"$\Delta\mu_{{{E1}}}$ (eV)", fontsize=13)
    ax.set_ylabel(rf"$\Delta\mu_{{{E2}}}$ (eV)", fontsize=13)
    ax.set_zlabel(rf"$\Delta\mu_{{{E3}}}$ (eV)", fontsize=13)
    ax.set_title(f"Chemical-potential feasible polyhedron in {E1}–{E2}–{E3}", fontsize=15)
    ax.grid(False)
    ax.view_init(elev=elev, azim=azim)
    plt.tight_layout()
    if out_png:
        plt.savefig(out_png, dpi=300)
        print(f"Saved {out_png}")
        plt.show()
    # ---------- optional export / return ----------
    data = None
    if export_csv:
        # Flatten faces to rows for simple inspection
        rows = []
        for fi, idxs in enumerate(faces):
            lab = face_labels[fi]
            for vi in idxs:
                rows.append({
                    "face_index": fi,
                    "face_label": lab,
                    "vx": V[vi, 0], "vy": V[vi, 1], "vz": V[vi, 2],
                    "vertex_index": vi
                })
        pd.DataFrame(rows).to_csv(export_csv, index=False)
        print(f"Exported vertices/faces to {export_csv}")
    if return_data:
        data = {
            "vertices": V,
            "faces": faces,
            "face_labels": face_labels,
            "A": A, "b": b, "labels": labels,
            "elements": elements,
            "floors": floors
        }
    return data

def plot_ternary(csv_path, elements, out_png=None, title=None,
                 label_mode="stable", target_names=None,
                 min_sep=0.035, fill_triangles=False):

    assert len(elements) == 3
    E1, E2, E3 = elements
    target_names = set(target_names or [])        # <-- your list: ["La2NiO4"], ["La2NiO4", "LaNiO3"], etc.

    df = pd.read_csv(csv_path)
    if 'Formation Energy (eV/atom)' in df.columns:
        col, to_pa = 'Formation Energy (eV/atom)', lambda E,N: float(E)
    elif 'Formation Energy (eV/fu)' in df.columns:
        col, to_pa = 'Formation Energy (eV/fu)', lambda E,N: float(E)/float(N)
    else:
        raise ValueError("CSV must have formation energy column")

    # === Collect compositions ===
    pts = []
    for f, val in zip(df['Formula'].astype(str), df[col]):
        c = _parse(f.strip())
        if not set(c).issubset(set(elements)): continue
        n1,n2,n3 = float(c.get(E1,0)), float(c.get(E2,0)), float(c.get(E3,0))
        N = n1+n2+n3
        if N <= 0: continue
        x1,x2,x3 = n1/N, n2/N, n3/N
        pts.append((x1,x2,x3,to_pa(val,N),f.strip()))

    # Add pure elements
    for comp,lab in [((1,0,0),E1),((0,1,0),E2),((0,0,1),E3)]:
        if not any(np.allclose([p[0],p[1],p[2]], comp, atol=1e-12) for p in pts):
            pts.append((*comp,0.0,lab))

    pts = np.array(pts, dtype=object)
    x1,x2,x3 = pts[:,0].astype(float), pts[:,1].astype(float), pts[:,2].astype(float)
    e = pts[:,3].astype(float)
    labels = pts[:,4].astype(str)

    # Deduplicate by composition
    key = np.round(np.c_[x1,x2,x3],10)
    uniq,inv = np.unique(key,axis=0,return_inverse=True)
    keep = [np.where(inv==ui)[0][np.argmin(e[np.where(inv==ui)])] for ui in range(len(uniq))]
    x1,x2,x3,e,labels = x1[keep],x2[keep],x3[keep],e[keep],labels[keep]

    # === Convex hull & stability ===
    P = np.c_[x1,x2,e]
    hull = ConvexHull(P, qhull_options="QJ")
    mix_idx = hull.vertices
    Xmix = np.c_[x1[mix_idx], x2[mix_idx], 1-x1[mix_idx]-x2[mix_idx]]
    Emix = e[mix_idx]

    def decomp_energy(i):
        Xi = np.array([x1[i], x2[i], 1-x1[i]-x2[i]])
        Aeq = np.vstack([Xmix.T, np.ones((1,len(mix_idx)))])
        beq = np.r_[Xi,1.0]
        res = linprog(c=Emix, A_eq=Aeq, b_eq=beq, bounds=(0,None), method="highs")
        return np.inf if not res.success else e[i] - res.fun

    de = np.array([decomp_energy(i) for i in range(len(e))])
    is_on_hull = de <= 1e-4
    lower_tris = [s for s,eq in zip(hull.simplices, hull.equations) if eq[2]<0]

    # Tie-lines
    tie_edges = set()
    for tri in lower_tris:
        i,j,k = tri
        for a,b in [(i,j),(j,k),(k,i)]:
            tie_edges.add(tuple(sorted([a,b])))
    stable_idx = set(i for edge in tie_edges for i in edge)

    # Barycentric → 2D
    h = np.sqrt(3)/2
    def bary_to_xy(x1,x2,x3):
        X = x2 + 0.5*x3
        Y = h*x3
        return X,Y
    X,Y = bary_to_xy(x1,x2,x3)

    # === Plotting ===
    fig, ax = plt.subplots(figsize=(8.6,7.6))
    ax.plot([0,1,0.5,0],[0,0,h,0],'k-',lw=2.5)

    if fill_triangles:
        for s in lower_tris:
            ax.fill(X[s], Y[s], color='0.85', alpha=0.6, zorder=1)

    for u,v in tie_edges:
        ax.plot([X[u],X[v]], [Y[u],Y[v]], 'k', lw=2.8, zorder=2)

    # Regular points (exclude targets)
    mask_target = np.isin(labels, list(target_names))
    mask_normal = ~mask_target

    ax.scatter(X[mask_normal & ~is_on_hull], Y[mask_normal & ~is_on_hull],
               s=26, c='lightgray', zorder=3)
    ax.scatter(X[mask_normal & is_on_hull], Y[mask_normal & is_on_hull],
               s=46, c='#86cc7a', edgecolor='k', lw=0.7, zorder=4)

    # TARGET PHASES → BIG RED DOT + optional label
    if mask_target.any():
        ax.scatter(X[mask_target], Y[mask_target],
                    c='red', edgecolor='darkred', lw=2.5, zorder=20)
        ax.scatter(X[mask_target], Y[mask_target],
                   c='red', alpha=0.25, zorder=19)  # soft glow

    # === Labeling (your existing greedy logic, unchanged) ===
    label_offset = {
        "LaNi5": (0.0, 0.02), "La2Ni7": (0.0, -0.03), "La7Ni6": (0.0, 0.08),
        "La7Ni16": (0.0, 0.04), "La7Ni3": (0.0, -0.02), "LaNi": (0.0, -0.02),
        "La2O3": (0.0, 0.03), "La2NiO4": (0.0, -0.01), "LaNiO3": (0.0, 0.04),
        "NiO": (0.04, 0.0), "Ni3O4": (0.04, 0.0)
    }

    def greedy_labels(mask, fontsize=12):
        placed = []
        r = 0.045
        cands = [(0,0),(r,0),(-r,0),(0,r),(0,-r),(r,r),(-r,r),(r,-r),(-r,-r)]
        for i in np.where(mask)[0]:
            xi,yi = X[i],Y[i]
            placed_ok = False
            for ox,oy in cands:
                lx,ly = xi+ox, yi+oy
                if not (-0.07<=lx<=1.07 and -0.07<=ly<=h+0.12): continue
                if placed and min(np.sum((np.array(placed)-[lx,ly])**2,axis=1)) < min_sep**2: continue
                dx,dy = label_offset.get(labels[i], (0,0))
                ax.text(lx+dx, ly+dy, labels[i], fontsize=fontsize,
                        ha='center', va='center', zorder=15, weight='bold')
                placed.append([lx,ly])
                placed_ok = True
                break
            if not placed_ok:
                ax.text(xi, yi+0.01, labels[i], fontsize=fontsize,
                        ha='center', va='bottom', zorder=15, weight='bold')

    if label_mode == "stable":
        mask = np.array([i in stable_idx for i in range(len(X))])
        mask |= np.isin(labels, list({E1,E2,E3} | target_names))
        greedy_labels(mask)

    # Corner labels
    ax.text(-0.03,-0.038,E1,fontsize=13,fontweight='bold',ha='right',va='top')
    ax.text(1.03,-0.038,E2,fontsize=13,fontweight='bold',ha='left',va='top')
    ax.text(0.50,h+0.045,E3,fontsize=13,fontweight='bold',ha='center',va='bottom')

    ax.set_aspect('equal','box')
    ax.set_xlim(-0.06,1.06); ax.set_ylim(-0.06,h+0.09)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title or f"{E1}–{E2}–{E3} Ternary Phase Diagram", fontsize=16)
    ax.grid(False)
    plt.tight_layout()

    if out_png:
        plt.savefig(out_png, dpi=400, bbox_inches='tight')
        print(f"Saved ternary diagram: {out_png}")
        plt.show()
        
def plot_mu_binary(csv_path, elements, mu_floor=-14.0, out_png=None, tol=1e-8):
    assert len(elements) == 2
    E1, E2 = elements

    df = pd.read_csv(csv_path)
    if 'Formation Energy (eV/fu)' not in df.columns:
        raise ValueError("Need 'Formation Energy (eV/fu)' column")

    # 1. Collect all phases with E1 or E2
    compounds = []
    for f, Ef in zip(df['Formula'], df['Formation Energy (eV/fu)']):
        c = _parse(f.strip())
        n1, n2 = float(c.get(E1, 0)), float(c.get(E2, 0))
        if n1 == 0 and n2 == 0: continue
        compounds.append((n1, n2, float(Ef), f.strip()))

    # 2. Deduplicate → lowest energy per reduced composition
    from math import gcd
    best = {}
    for n1, n2, E, lab in compounds:
        if n1 == 0:        key = (0, 1)
        elif n2 == 0:      key = (1, 0)
        else:
            a, b = int(round(n1*1e8)), int(round(n2*1e8))
            g = gcd(a, b)
            key = (a//g, b//g)
        if key not in best or E < best[key][0]:
            best[key] = (E, lab, n1, n2)

    # 3. Build inequality system + vertices
    A_real = np.array([[n1, n2] for _,_,n1,n2 in best.values()])
    b_real = np.array([E      for  E,_,_,_ in best.values()])
    labels =          [lab    for _,lab,_,_ in best.values()]

    A = np.vstack([A_real, [[1,0], [0,1], [-1,0], [0,-1]]])
    b = np.append(b_real, [0, 0, -mu_floor, -mu_floor])

    vertices = []
    for i in range(len(A)):
        for j in range(i+1, len(A)):
            if abs(np.linalg.det(A[[i,j]])) < tol: continue
            mu = np.linalg.solve(A[[i,j]], b[[i,j]])
            if np.all(A @ mu <= b + tol):
                vertices.append(mu)
    V = np.unique(np.round(vertices, 9), axis=0) if vertices else np.empty((0,2))

    # 4. Plotting
    fig, ax = plt.subplots(figsize=(11, 10.5))
    x_range = np.linspace(mu_floor, 0, 1300)

    # Draw all real phases
    for idx, (E, lab, n1, n2) in enumerate(best.values()):
        if lab == "La2NiO4":
            color, lw, alpha, z = 'red', 4.8, 1.0, 60
        else:
            color, lw, alpha, z = '#777777', 1.7, 0.75, 10

        if n2 == 0:
            x0 = E / n1
            if mu_floor <= x0 <= 0:
                ax.axvline(x0, color=color, lw=lw, alpha=alpha, zorder=z)
        elif n1 == 0:
            y0 = E / n2
            if mu_floor <= y0 <= 0:
                ax.axhline(y0, color=color, lw=lw, alpha=alpha, zorder=z)
        else:
            y = (E - n1 * x_range) / n2
            mask = (y >= mu_floor - 1e-6) & (y <= 1e-6)
            y[~mask] = np.nan
            ax.plot(x_range, y, color=color, lw=lw, alpha=alpha, zorder=z)

        # Label every real phase
        if n2 == 0 and mu_floor <= x0 <= 0:
            ax.annotate(lab, xy=(x0, mu_floor*0.6), xytext=(8,0), textcoords='offset points',
                       fontsize=10.5, color=color, fontweight='bold', ha='left',
                       bbox=dict(facecolor='white', alpha=0.92, edgecolor='none'))
        elif n1 == 0 and mu_floor <= y0 <= 0:
            ax.annotate(lab, xy=(mu_floor*0.6, y0), xytext=(0,8), textcoords='offset points',
                       fontsize=10.5, color=color, fontweight='bold', va='bottom',
                       bbox=dict(facecolor='white', alpha=0.92, edgecolor='none'))
        else:
            vis = np.where(mask)[0]
            if len(vis)>0:
                mid = vis[len(vis)//2]
                ax.annotate(lab, xy=(x_range[mid], y[mid]),
                           xytext=(10,10), textcoords='offset points',
                           fontsize=10.5, color=color, fontweight='bold',
                           bbox=dict(facecolor='white', alpha=0.94, edgecolor='none', pad=2.5),
                           arrowprops=dict(arrowstyle='-', color=color, lw=1.1, alpha=0.7))

    # 5. Stable region + label "Stable region"
    if len(V) >= 3:
        center = V.mean(axis=0)
        angles = np.arctan2(V[:,1]-center[1], V[:,0]-center[0])
        order = np.argsort(angles)
        V_ord = V[order]

        ax.fill(V_ord[:,0], V_ord[:,1], color='#e0e0e0', alpha=0.75, zorder=-2)
        ax.plot(V_ord[:,0], V_ord[:,1], 'k-', linewidth=1.5, zorder=-1)
        # Simple label inside the stable region
        ax.text(center[0], center[1], "Stable region",
                fontsize=10, color='black',
                ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='black', linewidth=1.0))

    # 6. Cosmetics
    ax.axvline(0, color='k', lw=1.8, alpha=0.8)
    ax.axhline(0, color='k', lw=1.8, alpha=0.8)
    ax.set_xlabel(rf'$\Delta\mu_{{{E1}}}$ (eV)', fontsize=16)
    ax.set_ylabel(rf'$\Delta\mu_{{{E2}}}$ (eV)', fontsize=16)
    ax.set_title(f'Chemical Potential Diagram – {E1}–{E2}', fontsize=18, pad=20)
    ax.set_xlim(mu_floor, 0); ax.set_ylim(mu_floor, 0)
    ax.grid(False)
    ax.set_aspect('equal', adjustable='box')

    if out_png:
        plt.savefig(out_png, dpi=450, bbox_inches='tight')
        print(f"Saved: {out_png}")
        plt.show()

CONFIG = {
    "material": "La2NiO4",
    "elements": ["La", "Ni", "O"],
    "defect_categories": ['Interstitial', 'Vacancy'],
    "defect_dir": ['Interstitial', 'Vacancy', 'Combined'],
    "e_above_hull": 0.09,
        "processes": 1,
}

def setup_directories(material, defect_dir):
    """Create directory structure and return paths"""
    base_dir = f"{material}_results"
    paths = {
        "base": base_dir,
        "dielectric": os.path.join(base_dir, "dielectric"),
        "chempot": os.path.join(base_dir, "chemical_potentials"),
        "defects": {cat: os.path.join(base_dir, cat) for cat in defect_dir},
    }
    for path in paths.values():
        if isinstance(path, dict):
            for p in path.values():
                os.makedirs(p, exist_ok=True)
        else:
            os.makedirs(path, exist_ok=True)
    return paths

if __name__ == "__main__":
    paths = setup_directories(CONFIG['material'], CONFIG['defect_dir'])
    csv_path = f"{paths['chempot']}/{CONFIG['material']}_formation_energies.csv"
    elements = ['La', 'Ni', 'O']
   
    element_pairs = [('La', 'Ni'), ('La', 'O'), ('Ni', 'O')]
    for pair in element_pairs:
        out_file1 = f"{paths['chempot']}/chemical_potential_{pair[0]}_{pair[1]}.png"
        plot_mu_binary(csv_path,
                       pair, mu_floor=-10.0, out_png=out_file1)
    # Create 3D diagram
    plot_ternary_polyhedron(csv_path, elements, out_png=f"{paths['chempot']}/chemical_potential_polyhedra.png")
    plot_ternary(csv_path, ["La", "Ni", "O"],target_names=["La2NiO4"], out_png=f"{paths['chempot']}/chemical_potential_ternary.png", label_mode="stable")