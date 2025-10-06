import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import math
import os
from PIL import Image
from matplotlib.animation import PillowWriter
import uuid
import io
import csv
import re

# Optional: for fetching 6-series presets (e.g., NACA 63-412)
try:
    import requests
except ImportError:
    requests = None

# Bokeh for interactive editors
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool, PointDrawTool, CustomJS
from bokeh.palettes import Category10

# Streamlit-Bokeh shim (Bokeh 3.x compatibility for Streamlit)
try:
    from streamlit_bokeh import streamlit_bokeh
except Exception:
    # Fallback to st.bokeh_chart (works if you're on Bokeh 2.4.3)
    def streamlit_bokeh(fig, **kwargs):
        st.bokeh_chart(fig, **kwargs)

# --------------------
# Bézier curve utilities
# --------------------
def bezier_curve(control_points, num_points=100):
    control_points = np.asarray(control_points, dtype=float)
    n = len(control_points) - 1
    t = np.linspace(0, 1, num_points)
    curve = np.zeros((num_points, 2), dtype=float)
    for i in range(n + 1):
        binomial = math.comb(n, i)
        bernstein = binomial * ((1 - t) ** (n - i)) * (t ** i)
        curve += np.outer(bernstein, control_points[i])
    return curve

def cubic_bezier(P0, P1, P2, P3, N=400):
    return bezier_curve([P0, P1, P2, P3], N)

def numeric_curvature(points):
    pts = np.asarray(points, float)
    dx = np.gradient(pts[:,0]); dy = np.gradient(pts[:,1])
    ddx = np.gradient(dx);      ddy = np.gradient(dy)
    denom = (dx*dx + dy*dy)**1.5 + 1e-12
    return np.abs(dx*ddy - dy*ddx) / denom

def poly_length(points):
    d = np.diff(np.asarray(points), axis=0)
    return np.sum(np.sqrt((d*d).sum(axis=1)))

# --------------------
# De Casteljau
# --------------------
def de_casteljau_levels(points, t):
    levels = [points]
    while len(levels[-1]) > 1:
        next_level = []
        for i in range(len(levels[-1]) - 1):
            p = (1 - t) * np.array(levels[-1][i]) + t * np.array(levels[-1][i + 1])
            next_level.append(p)
        levels.append(next_level)
    return levels

# --------------------
# Art helpers
# --------------------
def catmull_rom_to_bezier(points, closed=True):
    P = np.asarray(points, dtype=float)
    n = len(P)
    if n < 2: return []
    def get(idx): return P[idx % n] if closed else P[np.clip(idx, 0, n-1)]
    segs = []
    last = n if closed else (n - 1)
    for i in range(last):
        P0, P1, P2, P3 = get(i-1), get(i), get(i+1), get(i+2)
        B0 = P1
        B1 = P1 + (P2 - P0) / 6.0
        B2 = P2 - (P3 - P1) / 6.0
        B3 = P2
        segs.append(np.vstack([B0, B1, B2, B3]))
    return segs

def render_bezier_chain(segs, samples_per_seg=40):
    if not segs: return np.zeros((0,2))
    curves = [bezier_curve(seg, num_points=samples_per_seg) for seg in segs]
    out = [curves[0]]
    for c in curves[1:]:
        out.append(c[1:])
    return np.vstack(out)

def fill_closed_bezier(ax, segs, color, alpha=1.0, samples_per_seg=40, zorder=1):
    pts = render_bezier_chain(segs, samples_per_seg)
    if len(pts) > 2:
        ax.fill(pts[:,0], pts[:,1], color=color, alpha=alpha, zorder=zorder)

def save_current_figure(fig, filename):
    os.makedirs("gallery", exist_ok=True)
    fig.savefig(filename, dpi=180, bbox_inches="tight")

# --------------------
# Flower helpers
# --------------------
def draw_petal(ax, center, angle, length=1.0, width=0.5, color='purple', shape="Rounded"):
    if shape == "Rounded":
        p0 = np.array([0.0, 0.0]); p1 = np.array([width, length * 0.5]); p2 = np.array([0.0, length])
    elif shape == "Pointed":
        p0 = np.array([0.0, 0.0]); p1 = np.array([width, length]); p2 = np.array([0.0, length])
    elif shape == "Heart-shaped":
        p0 = np.array([0.0, 0.0]); p1 = np.array([width, length * 0.3]); p2 = np.array([0.0, length])
    else:
        p0 = np.array([0.0, 0.0]); p1 = np.array([width, length * 0.5]); p2 = np.array([0.0, length])
    curve1 = bezier_curve([p0, p1, p2])
    p1_mirror = np.array([-width, p1[1]])
    curve2 = bezier_curve([p2, p1_mirror, p0])
    petal = np.vstack((np.concatenate((curve1[:, 0], curve2[:, 0])),
                       np.concatenate((curve1[:, 1], curve2[:, 1]))))
    theta = np.radians(angle)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    petal = R @ petal + np.array(center)[:, None]
    ax.fill(petal[0], petal[1], color=color, alpha=0.6, edgecolor='black')

def draw_flower(num_petals=8, filename=None, colors=None, petal_length=2.0, petal_width=0.7, petal_shape="Rounded"):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal'); ax.axis('off')
    center = (0, 0)
    if colors is None:
        colors = cm.viridis(np.linspace(0, 1, num_petals))
    for i in range(num_petals):
        angle = i * (360 / num_petals)
        draw_petal(ax, center, angle, length=petal_length, width=petal_width, color=colors[i], shape=petal_shape)
    circle = plt.Circle(center, 0.3, color='gold', zorder=10); ax.add_patch(circle)
    plt.tight_layout()
    if filename:
        os.makedirs("gallery", exist_ok=True)
        plt.savefig(filename, dpi=150, bbox_inches='tight'); plt.close(fig)
    else:
        st.pyplot(fig)

# --------------------
# Shape-morph helpers
# --------------------
def circle_points(n_pts=60, radius=1.0):
    t = np.linspace(0, 2*np.pi, n_pts, endpoint=False)
    return np.column_stack((radius * np.cos(t), radius * np.sin(t)))

def ellipse_points(n_pts=60, a=1.2, b=0.8):
    t = np.linspace(0, 2*np.pi, n_pts, endpoint=False)
    return np.column_stack((a * np.cos(t), b * np.sin(t)))

def heart_points(n_pts=60):
    t = np.linspace(0, 2*np.pi, n_pts, endpoint=False)
    x = 16 * np.sin(t)**3
    y = 13*np.cos(t) - 5*np.cos(2*t) - 2*np.cos(3*t) - np.cos(4*t)
    pts = np.column_stack((x, y))
    r = np.sqrt((pts**2).sum(axis=1)).max()
    pts /= (r + 1e-9)
    return pts

def star_points(n_pts=60, r_outer=1.0, r_inner=0.45, spikes=5):
    t = np.linspace(0, 2*np.pi, n_pts, endpoint=False)
    idx = np.arange(n_pts) % (2*spikes)
    radii = np.where(idx < spikes, r_outer, r_inner)
    return np.column_stack((radii * np.cos(t), radii * np.sin(t)))

def polygon_points(n_pts=60, sides=6, radius=1.0, rotation=0.0):
    t = np.linspace(0, 2*np.pi, sides, endpoint=False) + rotation
    verts = np.column_stack((radius*np.cos(t), radius*np.sin(t)))
    return resample_polyline_closed(verts, n_pts)

def rose_points(n_pts=60, k=3, r=1.0):
    theta = np.linspace(0, 2*np.pi, n_pts, endpoint=False)
    rad = r * np.cos(k * theta)
    x = rad * np.cos(theta)
    y = rad * np.sin(theta)
    m = max(np.abs(x).max(), np.abs(y).max(), 1e-9)
    return np.column_stack((x/m, y/m))

def clover_points(n_pts=60, leaves=3):
    return rose_points(n_pts, k=leaves, r=1.0)

def resample_polyline_closed(points, n_pts):
    pts = np.asarray(points, dtype=float)
    segs = np.diff(np.vstack([pts, pts[0]]), axis=0)
    seglen = np.sqrt((segs**2).sum(axis=1))
    cum = np.insert(np.cumsum(seglen), 0, 0.0)
    total = cum[-1]
    if total < 1e-9:
        return np.tile(pts[0], (n_pts,1))
    s = np.linspace(0, total, n_pts+1)[:-1]
    out = []; j = 0
    for si in s:
        while not (cum[j] <= si <= cum[j+1]):
            j += 1
            if j >= len(pts):
                j = len(pts)-1; break
        t = (si - cum[j]) / max(cum[j+1]-cum[j], 1e-12)
        p = pts[j] + t * (pts[(j+1)%len(pts)] - pts[j])
        out.append(p)
    return np.array(out)

def polygon_to_cubic_segments(points):
    points = np.asarray(points, dtype=float)
    n = len(points); segs = []
    for i in range(n):
        p0 = points[i]; p3 = points[(i + 1) % n]; v = p3 - p0
        p1 = p0 + v / 3.0; p2 = p0 + 2.0 * v / 3.0
        segs.append(np.vstack([p0, p1, p2, p3]))
    return segs

def sample_bezier_segments(segments, samples_per_seg=20):
    curves = [bezier_curve(seg, num_points=samples_per_seg) for seg in segments]
    return np.vstack(curves)

def morph_segments(segments_a, segments_b, alpha):
    return [(1 - alpha) * np.asarray(sa) + alpha * np.asarray(sb) for sa, sb in zip(segments_a, segments_b)]

def best_cyclic_alignment(a, b):
    a = np.asarray(a); b = np.asarray(b); N = len(a)
    best_cost = np.inf; best_b = b.copy()
    for rev in [False, True]:
        bb = b[::-1].copy() if rev else b
        for shift in range(N):
            b_shift = np.roll(bb, shift, axis=0)
            cost = np.sum((a - b_shift)**2)
            if cost < best_cost:
                best_cost = cost; best_b = b_shift.copy()
    return best_b

def parse_points_text(text):
    pts = []
    for line in text.strip().splitlines():
        if not line.strip():
            continue
        parts = line.split(',') if ',' in line else line.split()
        if len(parts) >= 2:
            try:
                pts.append([float(parts[0]), float(parts[1])])
            except:
                continue
    return np.array(pts, dtype=float) if pts else None

def read_csv_points(file):
    try:
        content = file.read().decode('utf-8')
    except:
        content = file.read()
        if isinstance(content, bytes):
            content = content.decode('utf-8', errors='ignore')
    pts = []
    for row in csv.reader(io.StringIO(content)):
        if len(row) >= 2:
            try:
                pts.append([float(row[0]), float(row[1])])
            except:
                continue
    return np.array(pts, dtype=float) if pts else None

def normalize_points(pts):
    pts = np.asarray(pts, dtype=float)
    c = pts.mean(axis=0); pts = pts - c
    r = np.sqrt((pts**2).sum(axis=1)).max()
    if r > 1e-9:
        pts = pts / r
    return pts

# --------------------
# Airfoil helpers: NACA 4-digit generator, fitting, robust 6-series fetch
# --------------------
@st.cache_data(show_spinner=False)
def naca4_coords(code: str, n: int = 201):
    """
    Generate (xu,yu), (xl,yl) for NACA 4-digit, e.g. '0012', '2412'.
    """
    code = code.strip()
    if not re.fullmatch(r"\d{4}", code):
        raise ValueError("NACA 4-digit code must be 4 digits, e.g. 0012 or 2412")
    m = int(code[0]) / 100.0
    p = int(code[1]) / 10.0
    t = int(code[2:]) / 100.0

    beta = np.linspace(0, np.pi, n)
    x = 0.5*(1 - np.cos(beta))  # cosine spacing

    yt = 5*t*(0.2969*np.sqrt(np.maximum(x,0)) - 0.1260*x - 0.3516*x**2 + 0.2843*x**3 - 0.1015*x**4)

    yc = np.zeros_like(x)
    dyc_dx = np.zeros_like(x)
    with np.errstate(divide='ignore', invalid='ignore'):
        idx1 = x < p
        idx2 = ~idx1
        if p > 0:
            yc[idx1] = m/p**2 * (2*p*x[idx1] - x[idx1]**2)
            yc[idx2] = m/(1-p)**2 * ((1-2*p) + 2*p*x[idx2] - x[idx2]**2)
            dyc_dx[idx1] = 2*m/p**2 * (p - x[idx1])
            dyc_dx[idx2] = 2*m/(1-p)**2 * (p - x[idx2])

    theta = np.arctan(dyc_dx)
    xu = x - yt*np.sin(theta);  yu = yc + yt*np.cos(theta)
    xl = x + yt*np.sin(theta);  yl = yc - yt*np.cos(theta)

    oU = np.argsort(xu); oL = np.argsort(xl)
    return (xu[oU], yu[oU]), (xl[oL], yl[oL])

def _fit_cubic_one_side(x, y, P0, P3):
    x = np.asarray(x); y = np.asarray(y)
    x0, x1 = float(P0[0]), float(P3[0])
    denom = max(x1 - x0, 1e-9)
    t = np.clip((x - x0) / denom, 0.0, 1.0)

    C1 = 3*(1-t)**2 * t
    C2 = 3*(1-t) * t**2

    Bx = x - ((1-t)**3)*P0[0] - (t**3)*P3[0]
    By = y - ((1-t)**3)*P0[1] - (t**3)*P3[1]
    A = np.column_stack([C1, C2])

    P1x, P2x = np.linalg.lstsq(A, Bx, rcond=None)[0]
    P1y, P2y = np.linalg.lstsq(A, By, rcond=None)[0]
    P1 = np.array([P1x, P1y]); P2 = np.array([P2x, P2y])
    return P1, P2

def fit_cubic_bezier_to_airfoil(xu, yu, xl, yl):
    xmin = min(xu.min(), xl.min()); xmax = max(xu.max(), xl.max())
    scale = xmax - xmin
    xuN = (xu - xmin) / max(scale, 1e-9); xlN = (xl - xmin) / max(scale, 1e-9)
    yuN = yu.copy(); ylN = yl.copy()

    i_le_u = np.argmin(np.abs(xuN - 0.0)); i_te_u = np.argmin(np.abs(xuN - 1.0))
    i_le_l = np.argmin(np.abs(xlN - 0.0)); i_te_l = np.argmin(np.abs(xlN - 1.0))

    P0u = np.array([xuN[i_le_u], yuN[i_le_u]])
    P3u = np.array([xuN[i_te_u], yuN[i_te_u]])
    P0l = np.array([xlN[i_le_l], ylN[i_le_l]])
    P3l = np.array([xlN[i_te_l], ylN[i_te_l]])

    P1u, P2u = _fit_cubic_one_side(xuN, yuN, P0u, P3u)
    P1l, P2l = _fit_cubic_one_side(xlN, ylN, P0l, P3l)
    return (P0u, P1u, P2u, P3u), (P0l, P1l, P2l, P3l)

@st.cache_data(show_spinner=True)
def fetch_airfoil_coords_from_web(name: str):
    """
    Robust fetcher for 6-series like 'NACA 63-412', 'NACA 64-212', etc.
    Tries multiple slug variants on UIUC + AirfoilTools.
    """
    if requests is None:
        return None

    base = name.strip()
    raw = base.lower().replace(" ", "")
    raw_nohy = raw.replace("-", "")
    raw_caps = raw.upper().replace("-", "")
    slugs = {raw, raw_nohy, raw_caps, raw.replace("-", ""), base.replace(" ", ""), base.replace("-", ""), base}

    candidates = []
    for slug in slugs:
        candidates += [
            f"https://m-selig.ae.illinois.edu/ads/coord/{slug}.dat",
            f"https://m-selig.ae.illinois.edu/ads/coord/{slug.lower()}.dat",
            f"https://m-selig.ae.illinois.edu/ads/coord/{slug.upper()}.dat",
        ]
        hy = re.sub(r"(?i)naca(\d+)-?(\d+)", r"naca\1-\2", slug.lower())
        candidates.append(f"https://m-selig.ae.illinois.edu/ads/coord/{hy}.dat")
        candidates += [
            f"https://airfoiltools.com/airfoil/seligdatfile?airfoil={slug}",
            f"https://airfoiltools.com/airfoil/seligdatfile?airfoil={slug}-il",
            f"https://airfoiltools.com/airfoil/seligdatfile?airfoil={hy}",
            f"https://airfoiltools.com/airfoil/seligdatfile?airfoil={hy}-il",
        ]

    for url in candidates:
        try:
            r = requests.get(url, timeout=10)
            if r.status_code != 200 or len(r.text) < 500:
                continue
            txt = r.text
            lines = [ln.strip() for ln in txt.splitlines()]
            pts = []
            for ln in lines:
                if not ln: continue
                parts = re.split(r"[,\s]+", ln)
                if len(parts) >= 2:
                    try:
                        x = float(parts[0]); y = float(parts[1])
                        pts.append((x, y))
                    except:
                        pass
            if len(pts) < 40:
                continue
            pts = np.array(pts, float)

            xs = pts[:, 0]
            xmin, xmax = xs.min(), xs.max()
            chord = max(xmax - xmin, 1e-9)
            xsN = (xs - xmin) / chord

            xg = np.linspace(0, 1, 601)
            yu = np.full_like(xg, np.nan, float)
            yl = np.full_like(xg, np.nan, float)
            for i, xv in enumerate(xg):
                mask = np.abs(xsN - xv) < 0.01
                if not np.any(mask):
                    continue
                ys = pts[mask, 1]
                yu[i] = np.nanmax(ys)
                yl[i] = np.nanmin(ys)

            def _fill(z):
                idx = ~np.isnan(z)
                return np.interp(xg, xg[idx], z[idx])
            yu = _fill(yu); yl = _fill(yl)
            return (xg, yu), (xg, yl)
        except Exception:
            continue
    return None

# --------------------
# Streamlit UI
# --------------------
st.set_page_config(page_title="Bézier Curve Generator", layout="wide")
st.title("Bézier Curve Generator")

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    ["Interactive Bézier Curve",
     "Animated Bézier Curve",
     "Flower Gallery",
     "Shape Morphing",
     "Bézier Art Showcase",
     "Airfoil Designer",
     "Path Planning (Optimal)"]
)

# --------------------
# Tab 1: Interactive Bézier (Bokeh)
# --------------------
with tab1:
    st.header("Define Your Own Bézier Curve")
    n_points = st.slider("Number of control points", 2, 6, 3, key="t1_n")
    control_points = []
    for i in range(n_points):
        cols = st.columns(2)
        x = cols[0].number_input(f"Point {i+1} - x", value=float(i * 2), step=0.5, key=f"t1_x{i}")
        y = cols[1].number_input(f"Point {i+1} - y", value=float(i), step=0.5, key=f"t1_y{i}")
        control_points.append([x, y])

    curve_np = bezier_curve(control_points, num_points=200)
    cp = np.array(control_points)
    cp_source = ColumnDataSource(data=dict(x=cp[:,0].tolist(), y=cp[:,1].tolist()))
    cp_line_source = ColumnDataSource(data=dict(x=cp[:,0].tolist(), y=cp[:,1].tolist()))
    curve_source = ColumnDataSource(data=dict(x=curve_np[:,0].tolist(), y=curve_np[:,1].tolist()))

    p = figure(height=500, title="Bézier Curve (drag the red points)",
               tools="pan,wheel_zoom,reset,save,hover", match_aspect=True)
    p.add_tools(HoverTool(tooltips=[("x", "@x{0.00}"), ("y", "@y{0.00}")]))
    p.grid.visible = True
    p.line('x', 'y', source=curve_source, line_width=3, line_color="#1f77b4", legend_label="Bézier Curve")
    p.line('x', 'y', source=cp_line_source, line_dash="dashed", line_color="#888", legend_label="Control Polygon")
    cp_renderer = p.circle('x', 'y', source=cp_source, size=10, color="crimson", line_color="black", legend_label="Control Points")
    draw_tool = PointDrawTool(renderers=[cp_renderer], add=False)
    p.add_tools(draw_tool); p.toolbar.active_tap = draw_tool

    callback_tab1 = CustomJS(args=dict(cp_source=cp_source, cp_line_source=cp_line_source, curve_source=curve_source), code="""
        const xs = cp_source.data['x'], ys = cp_source.data['y']; const n = xs.length - 1;
        function comb(n,k){ if(k<0||k>n) return 0; if(k===0||k===n) return 1; let r=1; for(let i=1;i<=k;i++){ r=r*(n-(k-i))/i;} return r;}
        const N=200, cx=new Array(N), cy=new Array(N);
        for(let j=0;j<N;j++){ const t=j/(N-1); let bx=0, by=0;
            for(let i=0;i<=n;i++){ const b=comb(n,i)*Math.pow(1-t,n-i)*Math.pow(t,i); bx+=b*xs[i]; by+=b*ys[i]; }
            cx[j]=bx; cy[j]=by; }
        curve_source.data={x:cx,y:cy}; cp_line_source.data={x:xs,y:ys};
        cp_source.change.emit(); curve_source.change.emit(); cp_line_source.change.emit();
    """)
    cp_source.js_on_change('data', callback_tab1)

    p.x_range.start = min(cp[:,0].min(), curve_np[:,0].min()) - 1
    p.x_range.end   = max(cp[:,0].max(), curve_np[:,0].max()) + 1
    p.y_range.start = min(cp[:,1].min(), curve_np[:,1].min()) - 1
    p.y_range.end   = max(cp[:,1].max(), curve_np[:,1].max()) + 1
    p.legend.click_policy = "hide"
    streamlit_bokeh(p, use_container_width=True, key="tab1_bezier")

# --------------------
# Tab 2: De Casteljau demo (Bokeh)
# --------------------
with tab2:
    st.header("Interactive Bézier Curve Formation with Slider (Bokeh)")
    n_anim_points = st.slider("Number of control points (3–10)", 3, 10, 4, key="t2_n")
    anim_control_points = []
    for i in range(n_anim_points):
        cols = st.columns(2)
        x = cols[0].number_input(f"Point {i+1} - x", value=float(i), step=0.5, key=f"t2_x{i}")
        y = cols[1].number_input(f"Point {i+1} - y", value=float(i % 2), step=0.5, key=f"t2_y{i}")
        anim_control_points.append([x, y])
    t0 = float(st.slider("Parameter t", 0.0, 1.0, 0.0, 0.01, key="t2_t"))
    cp = np.array(anim_control_points)
    levels_py = de_casteljau_levels(anim_control_points, t0)

    def bezier_point(points, tau):
        L = [np.array(p, dtype=float) for p in points]
        while len(L) > 1:
            L = [(1-tau)*L[i] + tau*L[i+1] for i in range(len(L)-1)]
        return L[0]
    steps = max(2, int(max(t0, 1e-9)*200))
    taus = np.linspace(0.0, max(t0, 1e-9), steps)
    pc = np.array([bezier_point(anim_control_points, tau) for tau in taus])

    level_sources, palette = [], Category10[10]
    for lvl in levels_py[:-1]:
        pts = np.array(lvl)
        level_sources.append(ColumnDataSource(data=dict(x=pts[:,0].tolist(), y=pts[:,1].tolist())))

    partial_source = ColumnDataSource(data=dict(x=pc[:,0].tolist(), y=pc[:,1].tolist()))
    point_source   = ColumnDataSource(data=dict(x=[pc[-1,0]], y=[pc[-1,1]]))
    cp_source2     = ColumnDataSource(data=dict(x=cp[:,0].tolist(), y=cp[:,1].tolist()))
    cp_line_source = ColumnDataSource(data=dict(x=cp[:,0].tolist(), y=cp[:,1].tolist()))

    p2 = figure(height=500, title="De Casteljau's Algorithm",
                tools="pan,wheel_zoom,reset,save,hover", match_aspect=True)
    p2.add_tools(HoverTool(tooltips=[("x", "@x{0.00}"), ("y", "@y{0.00}")]))
    p2.grid.visible = True
    p2.line('x', 'y', source=cp_line_source, color="#666", line_dash="dashed", legend_label="Control Polygon")
    cp_renderer = p2.circle('x', 'y', source=cp_source2, size=9, color="#d62728", line_color="black", legend_label="Control Points")
    for i, src in enumerate(level_sources):
        p2.line('x', 'y', source=src, color=palette[i % len(palette)], line_width=2, legend_label=f"Level {i+1}")
        p2.circle('x', 'y', source=src, color=palette[i % len(palette)], size=6)
    p2.line('x', 'y', source=partial_source, color="#e41a1c", line_width=3, legend_label="Partial Bézier Curve")
    p2.circle('x', 'y', source=point_source, size=10, color="#1f77b4", legend_label="Bézier Point")
    draw_tool2 = PointDrawTool(renderers=[cp_renderer], add=False)
    p2.add_tools(draw_tool2); p2.toolbar.active_tap = draw_tool2

    callback_tab2 = CustomJS(args=dict(
        cp_source=cp_source2, cp_line_source=cp_line_source, level_sources=level_sources,
        partial_source=partial_source, point_source=point_source, tval=t0
    ), code="""
        const xs0 = cp_source.data['x']; const ys0 = cp_source.data['y']; const t = tval;
        cp_line_source.data = {x: xs0, y: ys0};
        let X = xs0.slice(), Y = ys0.slice(); const allX=[X.slice()], allY=[Y.slice()];
        while (X.length > 1) {
            const Xn=[], Yn=[];
            for (let i=0;i<X.length-1;i++){ Xn.push((1-t)*X[i] + t*X[i+1]); Yn.push((1-t)*Y[i] + t*Y[i+1]); }
            X=Xn; Y=Yn; allX.push(X.slice()); allY.push(Y.slice());
        }
        for (let k=0;k<level_sources.length;k++){
            const d=(k<allX.length-1)?{x:allX[k],y:allY[k]}:{x:[],y:[]};
            level_sources[k].data=d; level_sources[k].change.emit();
        }
        const bx=allX[allX.length-1][0], by=allY[allY.length-1][0];
        point_source.data={x:[bx],y:[by]};
        const steps=Math.max(2, Math.floor(t*200));
        const pcx=[], pcy=[];
        for (let j=0;j<steps;j++){
            const tau=(steps===1)?t: j/(steps-1)*t;
            let Lx=xs0.slice(), Ly=ys0.slice();
            while (Lx.length>1){
                const nx=[], ny=[];
                for (let i=0;i<Lx.length-1;i++){
                    nx.push((1-tau)*Lx[i] + tau*Lx[i+1]);
                    ny.push((1-tau)*Ly[i] + tau*Ly[i+1]);
                }
                Lx=nx; Ly=ny;
            }
            pcx.push(Lx[0]); pcy.push(Ly[0]);
        }
        partial_source.data={x:pcx,y:pcy};
        cp_source.change.emit(); cp_line_source.change.emit();
        point_source.change.emit(); partial_source.change.emit();
    """)
    cp_source2.js_on_change('data', callback_tab2)
    p2.x_range.start = min(cp[:,0].min(), -1) - 1
    p2.x_range.end   = max(cp[:,0].max(),  1) + 1
    p2.y_range.start = min(cp[:,1].min(), -1) - 1
    p2.y_range.end   = max(cp[:,1].max(),  1) + 1
    p2.legend.click_policy = "hide"
    streamlit_bokeh(p2, use_container_width=True, key="tab2_casteljau")

# --------------------
# Tab 3: Flower Gallery
# --------------------
with tab3:
    st.header("Bézier Flower Generator & Gallery")
    cols = st.columns([1, 2])
    with cols[0]:
        st.subheader("Generate a New Flower")
        flower_name = st.text_input("Flower Name", value="MyFlower", key="fl_name")
        petals = st.slider("Number of Petals", 3, 20, 6, key="fl_petals")
        petal_length = st.slider("Petal Length", 0.5, 4.0, 2.0, step=0.1, key="fl_plen")
        petal_width = st.slider("Petal Width", 0.1, 2.0, 0.7, step=0.1, key="fl_pwid")
        petal_shape = st.selectbox("Petal Shape", ["Rounded", "Pointed", "Heart-shaped"], key="fl_shape")
        color_mode = st.radio("Petal Color Mode", ["Autogenerated", "Single Color", "Custom Each Petal"], key="fl_colmode")
        if color_mode == "Single Color":
            single_color = st.color_picker("Pick Petal Color", "#FF69B4", key="fl_single_color")
            petal_colors = [single_color] * petals
        elif color_mode == "Custom Each Petal":
            petal_colors = [st.color_picker(f"Petal {i+1} Color", "#FF69B4", key=f"fl_petal_color_{i}") for i in range(petals)]
        else:
            petal_colors = cm.viridis(np.linspace(0, 1, petals))
        generate = st.button("🌼 Generate and Save Flower", key="fl_gen")
    if generate:
        filename = f"gallery/{flower_name}_{uuid.uuid4().hex[:6]}.png"
        draw_flower(petals, filename, petal_colors, petal_length, petal_width, petal_shape)
        st.success(f"Saved as {filename}")
    gallery_dir = "gallery"
    if os.path.exists(gallery_dir):
        image_files = sorted([os.path.join(gallery_dir, f) for f in os.listdir(gallery_dir) if f.endswith(".png")])
        if image_files:
            st.subheader("🖼️ Your Saved Flowers")
            gallery_cols = st.columns(3)
            for i, image_file in enumerate(image_files):
                with gallery_cols[i % 3]:
                    st.image(Image.open(image_file), caption=os.path.basename(image_file), use_container_width=True)
                    if st.button(f"🗑 Delete {os.path.basename(image_file)}", key=f"fl_del_{i}"):
                        os.remove(image_file); st.rerun()
        else:
            st.info("No flower images found in the gallery yet.")
    else:
        st.info("Generate a flower to create the gallery folder.")

# --------------------
# Tab 4: Shape Morphing
# --------------------
with tab4:
    st.header("Shape Morphing with Bézier Curves")
    st.info("Morph between two closed shapes via Bézier control-point interpolation.")
    n_segments = st.slider("Number of segments (higher = smoother)", 12, 200, 80, step=4, key="m_nseg")
    samples_per_seg = st.slider("Samples per segment (render quality)", 5, 80, 24, step=1, key="m_sps")
    st.subheader("Preset Shapes")
    preset_col = st.columns(2)
    shape_builders = {
        "Circle":           lambda n: circle_points(n),
        "Ellipse":          lambda n: ellipse_points(n, a=1.3, b=0.9),
        "Heart":            lambda n: heart_points(n),
        "Star (5 spikes)":  lambda n: star_points(n, spikes=5),
        "Star (7 spikes)":  lambda n: star_points(n, spikes=7),
        "Triangle":         lambda n: polygon_points(n, sides=3, radius=1.0, rotation=np.pi/2),
        "Square":           lambda n: polygon_points(n, sides=4, radius=1.0, rotation=np.pi/4),
        "Pentagon":         lambda n: polygon_points(n, sides=5, radius=1.0),
        "Hexagon":          lambda n: polygon_points(n, sides=6, radius=1.0),
        "Clover (3)":       lambda n: clover_points(n, leaves=3),
        "Clover (4)":       lambda n: clover_points(n, leaves=4),
        "Rose (5)":         lambda n: rose_points(n, k=5),
    }
    start_shape = preset_col[0].selectbox("Start Preset", list(shape_builders.keys()), index=0, key="m_start")
    end_shape   = preset_col[1].selectbox("End Preset",   list(shape_builders.keys()), index=2, key="m_end")
    use_presets = st.checkbox("Use presets above", value=True, key="m_usepresets")

    st.subheader("Custom Shapes (optional)")
    c1, c2 = st.columns(2)
    with c1:
        st.caption("Custom Shape A")
        txt_a = st.text_area("Paste points for A (x,y per line or 'x y')", height=120, key="m_txt_a")
        file_a = st.file_uploader("…or upload CSV for A (two columns: x,y)", type=["csv"], key="m_file_a")
    with c2:
        st.caption("Custom Shape B")
        txt_b = st.text_area("Paste points for B (x,y per line or 'x y')", height=120, key="m_txt_b")
        file_b = st.file_uploader("…or upload CSV for B (two columns: x,y)", type=["csv"], key="m_file_b")

    if use_presets:
        poly_a = shape_builders[start_shape](n_segments)
        poly_b = shape_builders[end_shape](n_segments)
    else:
        pts_a = read_csv_points(file_a) if file_a is not None else (parse_points_text(txt_a) if txt_a.strip() else None)
        pts_b = read_csv_points(file_b) if file_b is not None else (parse_points_text(txt_b) if txt_b.strip() else None)
        if (pts_a is None) or (pts_b is None) or (len(pts_a) < 3) or (len(pts_b) < 3):
            st.warning("Provide valid custom points for both A and B (≥3 points each). Falling back to presets.")
            poly_a = shape_builders[start_shape](n_segments)
            poly_b = shape_builders[end_shape](n_segments)
        else:
            pts_a = normalize_points(pts_a); pts_b = normalize_points(pts_b)
            poly_a = resample_polyline_closed(pts_a, n_segments)
            poly_b = resample_polyline_closed(pts_b, n_segments)

    align = st.checkbox("Auto-align shapes (cyclic shift & direction)", value=True, key="m_align")
    if align: poly_b = best_cyclic_alignment(poly_a, poly_b)
    morph_t = st.slider("Morph Progress", 0.0, 1.0, 0.5, step=0.01, key="m_t")

    segs_a = polygon_to_cubic_segments(poly_a)
    segs_b = polygon_to_cubic_segments(poly_b)
    segs_m = morph_segments(segs_a, segs_b, morph_t)

    start_curve = sample_bezier_segments(segs_a, samples_per_seg=samples_per_seg)
    end_curve   = sample_bezier_segments(segs_b, samples_per_seg=samples_per_seg)
    morph_curve = sample_bezier_segments(segs_m, samples_per_seg=samples_per_seg)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(start_curve[:, 0], start_curve[:, 1], 'g--', alpha=0.5, label="Start")
    ax.plot(end_curve[:, 0],   end_curve[:, 1],   'b--', alpha=0.5, label="End")
    ax.plot(morph_curve[:, 0], morph_curve[:, 1], 'r-',  linewidth=2, label="Morph")
    if st.checkbox("Show control polygons (first 3 segments)", key="m_ctrl"):
        for seg in segs_m[:3]:
            seg = np.asarray(seg); ax.plot(seg[:, 0], seg[:, 1], 'ro--', alpha=0.6)
    ax.set_aspect('equal'); ax.axis('off'); ax.legend()
    st.pyplot(fig)

# --------------------
# Tab 5: Art Showcase (Galaxy Streamers + FAST Topo City Skyline)
# --------------------
with tab5:
    st.header("Bézier Art Showcase")
    st.write("Pick a style, tweak parameters, and export as PNG.")

    mode = st.radio(
        "Choose a drawing",
        ["Mountains & Trees", "Abstract Curves", "Galaxy Streamers", "Topo City Skyline"],
        horizontal=True,
        key="art_mode"
    )

    # 1) Mountains & Trees
    if mode == "Mountains & Trees":
        cols = st.columns(4)
        with cols[0]: layers = st.slider("Mountain layers", 2, 6, 4, key="art_layers")
        with cols[1]: tree_count = st.slider("Trees", 5, 60, 24, key="art_trees")
        with cols[2]: snow_ratio = st.slider("Snow cap ratio", 0.0, 0.5, 0.18, step=0.02, key="art_snow")
        with cols[3]: seed = st.number_input("Random seed", value=42, step=1, key="seed_mountains")
        np.random.seed(int(seed))

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_aspect('equal'); ax.axis('off')
        ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.1, 1.2)

        # Sky brushy bands
        for k in range(6):
            y0 = 0.2 + 0.12*k
            anchors = np.array([[-1.7, y0], [-1.0, y0+0.02], [0.0, y0+0.04], [1.0, y0+0.02], [1.7, y0]])
            segs = catmull_rom_to_bezier(anchors, closed=False)
            pts = render_bezier_chain(segs, 120)
            ax.plot(pts[:,0], pts[:,1], alpha=0.08, linewidth=18)

        base_colors = ["#243b55", "#2b5876", "#4e4376", "#6a3093", "#355c7d", "#2c3e50"]
        for i in range(layers):
            y_base = -1.0 + i*0.35
            width  = 1.6 - i*0.12
            peaks  = np.linspace(-width, width, 6 + i) + np.random.uniform(-0.12, 0.12, 6 + i)
            heights = y_base + np.random.uniform(0.15, 0.42, len(peaks))
            anchors = np.column_stack([peaks, heights])
            anchors = np.vstack([[-1.7, y_base-0.05], anchors, [1.7, y_base-0.05]])
            segs = catmull_rom_to_bezier(anchors, closed=True)
            fill_closed_bezier(ax, segs, color=base_colors[i % len(base_colors)], alpha=0.40, samples_per_seg=80, zorder=2+i)

            # Snow caps
            if snow_ratio > 0 and i >= layers-2:
                ridge = anchors[1:-1]
                top = ridge[np.argmax(ridge[:,1])]
                cap = np.array([
                    top + [-0.18, -snow_ratio*0.2],
                    top + [-0.06,  snow_ratio*0.05],
                    top + [ 0.06,  snow_ratio*0.05],
                    top + [ 0.18, -snow_ratio*0.2],
                ])
                segs_cap = catmull_rom_to_bezier(cap, closed=False)
                pts_cap = render_bezier_chain(segs_cap, 60)
                ax.plot(pts_cap[:,0], pts_cap[:,1], color="#f1f2f6", linewidth=3, alpha=0.8, zorder=3+i)

        # Trees
        def draw_tree(x, y, scale=1.0, z=10):
            trunk_h = 0.2 * scale
            trunk_w_bot = 0.035 * scale
            trunk_w_top = 0.025 * scale
            trunk = np.array([
                [x - trunk_w_bot, y - trunk_h],
                [x + trunk_w_bot, y - trunk_h],
                [x + trunk_w_top, y],
                [x - trunk_w_top, y]
            ])
            segs_trunk = catmull_rom_to_bezier(trunk, closed=True)
            fill_closed_bezier(ax, segs_trunk, "#5b3a29", alpha=0.95, samples_per_seg=16, zorder=z)

            layers_local = 3
            layer_h = 0.18 * scale
            base_w = 0.26 * scale
            green_shades = ["#2e8b57", "#3b9d65", "#276b48"]

            for j in range(layers_local):
                top_y = y + 0.02*scale + j*(layer_h*0.6)
                bottom_y = top_y - layer_h
                width_top = base_w * (1 - 0.3*j) * 0.5
                width_bottom = base_w * (1 - 0.3*j)
                jitter = 0.02 * scale
                left_skew = np.random.uniform(-0.015, 0.015) * scale
                right_skew = np.random.uniform(-0.015, 0.015) * scale

                foliage = np.array([
                    [x + np.random.uniform(-jitter, jitter),                   top_y + np.random.uniform(-jitter, jitter)],
                    [x - width_top + left_skew,                                top_y - layer_h*0.3],
                    [x - width_bottom + left_skew,                             bottom_y + np.random.uniform(-jitter, jitter)],
                    [x + np.random.uniform(-0.01*scale, 0.01*scale),           bottom_y - 0.02*scale],
                    [x + width_bottom + right_skew,                            bottom_y + np.random.uniform(-jitter, jitter)],
                    [x + width_top + right_skew,                               top_y - layer_h*0.3]
                ])

                segs_foliage = catmull_rom_to_bezier(foliage, closed=True)
                color = green_shades[j % len(green_shades)]
                fill_closed_bezier(ax, segs_foliage, color, alpha=0.88, samples_per_seg=40, zorder=z + j + 1)

        xs = np.random.uniform(-1.4, 1.4, tree_count)
        ys = np.random.uniform(-0.95, 0.8, tree_count)
        scales = np.clip(np.random.normal(1.0, 0.35, tree_count), 0.5, 1.8)
        order = np.argsort(ys)
        for idx in order:
            draw_tree(xs[idx], ys[idx], scales[idx], z=20+idx)

        colx = st.columns(2)
        with colx[0]:
            if st.button("💾 Save PNG (Mountains & Trees)", key="art_save_mtn"):
                fname = f"gallery/bezier_mountains_{uuid.uuid4().hex[:6]}.png"
                save_current_figure(fig, fname); st.success(f"Saved: {fname}")

        st.pyplot(fig)

    # 2) Abstract Curves
    elif mode == "Abstract Curves":
        cols = st.columns(4)
        with cols[0]: n_curves = st.slider("Number of curves", 5, 120, 40, step=1, key="art_ncurves")
        with cols[1]: jitter = st.slider("Curvature randomness", 0.0, 1.0, 0.55, step=0.05, key="art_jitter")
        with cols[2]: thickness = st.slider("Max stroke width", 1.0, 12.0, 5.0, step=0.5, key="art_thick")
        with cols[3]: seed = st.number_input("Random seed", value=7, step=1, key="seed_abstract")

        np.random.seed(int(seed))
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.set_aspect('equal'); ax.axis('off')
        ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2)

        def hsv_to_rgb(h, s, v):
            return tuple(float(x) for x in plt.cm.hsv(h % 1.0)[:3])

        for i in range(n_curves):
            m = np.random.randint(4, 7)
            pts = np.random.uniform(-1.0, 1.0, (m, 2))
            pts *= (0.5 + 0.5*np.random.rand()) * (0.65 + 0.35*(1.0 - jitter))
            pts += np.random.normal(0, 0.12*jitter, pts.shape)

            closed = (np.random.rand() < 0.4)
            segs = catmull_rom_to_bezier(pts, closed=closed)
            curve = render_bezier_chain(segs, samples_per_seg=120)

            color = hsv_to_rgb(i / max(1, n_curves), 0.85, 0.9)
            lw = np.random.uniform(0.6, thickness)
            ax.plot(curve[:,0], curve[:,1], color=color, linewidth=lw, alpha=0.9)

            if closed and (np.random.rand() < 0.35):
                ax.fill(curve[:,0], curve[:,1], color=color, alpha=0.12)

        colx = st.columns(2)
        with colx[0]:
            if st.button("💾 Save PNG (Abstract)", key="art_save_abs"):
                fname = f"gallery/bezier_abstract_{uuid.uuid4().hex[:6]}.png"
                save_current_figure(fig, fname); st.success(f"Saved: {fname}")

        st.pyplot(fig)

    # 3) Galaxy Streamers
    elif mode == "Galaxy Streamers":
        g = st.columns(5)
        with g[0]: arms = st.slider("Arms", 1, 8, 4, key="gal_arms")
        with g[1]: turns = st.slider("Turns", 1.0, 6.0, 3.5, 0.1, key="gal_turns")
        with g[2]: growth = st.slider("Spiral growth (b)", 0.10, 0.50, 0.28, 0.01, key="gal_growth")
        with g[3]: jitter = st.slider("Jitter", 0.0, 0.5, 0.12, 0.01, key="gal_jitter")
        with g[4]: seed = st.number_input("Seed", value=11, step=1, key="gal_seed")

        d = st.columns(4)
        with d[0]: nodes = st.slider("Nodes per arm", 80, 800, 300, 20, key="gal_nodes")
        with d[1]: sps   = st.slider("Samples/segment", 30, 200, 100, 10, key="gal_sps")
        with d[2]: lw    = st.slider("Line width", 0.5, 6.0, 2.6, 0.1, key="gal_lw")
        with d[3]: stars = st.slider("Star count", 0, 2000, 600, 50, key="gal_stars")

        np.random.seed(int(seed))
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.set_aspect('equal'); ax.axis('off')

        theta_max = turns * 2*np.pi
        a = 0.05
        colors = plt.cm.magma(np.linspace(0.15, 0.95, arms))

        for k in range(arms):
            theta = np.linspace(0.0, theta_max, nodes)
            theta += (2*np.pi*k)/max(1, arms)
            r = a * np.exp(growth * (theta - theta[0]))
            x = r * np.cos(theta); y = r * np.sin(theta)

            dx = -np.sin(theta); dy = np.cos(theta)
            mag = jitter * (0.2 + 0.8*np.linspace(0,1,nodes))
            x += dx * (mag * (np.random.rand(nodes)-0.5))
            y += dy * (mag * (np.random.rand(nodes)-0.5))

            pts = np.column_stack([x, y])
            m = np.max(np.sqrt((pts**2).sum(axis=1))) + 1e-9
            pts /= (m/1.1)

            segs = catmull_rom_to_bezier(pts, closed=False)
            curve = render_bezier_chain(segs, samples_per_seg=int(sps))
            ax.plot(curve[:,0], curve[:,1], color=colors[k], lw=lw, alpha=0.9)

        core = np.linspace(0, 2*np.pi, 180, endpoint=False)
        ax.scatter(0.03*np.cos(core), 0.03*np.sin(core), s=10, c="#ffd27d", alpha=0.25, linewidths=0)

        if stars > 0:
            sx = np.random.uniform(-1.3, 1.3, stars)
            sy = np.random.uniform(-1.3, 1.3, stars)
            r = np.sqrt(sx**2 + sy**2)
            mask = (r > 0.3)
            sizes = np.random.choice([2,3,4], size=stars)
            ax.scatter(sx[mask], sy[mask], s=sizes[mask], c="white", alpha=0.6, linewidths=0)

        ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4)

        if st.button("💾 Save PNG (Galaxy Streamers)", key="gal_save"):
            fname = f"gallery/bezier_galaxy_{uuid.uuid4().hex[:6]}.png"
            save_current_figure(fig, fname); st.success(f"Saved: {fname}")

        st.pyplot(fig)

    # 4) Topo City Skyline (FAST)
    else:
        # Controls (FAST)
        c = st.columns(6)
        with c[0]: buildings = st.slider("Buildings", 10, 100, 40, 1, key="sky_buildings")
        with c[1]: rough     = st.slider("Roughness", 0.0, 0.6, 0.25, 0.01, key="sky_rough")
        with c[2]: layers    = st.slider("Contour layers", 5, 60, 18, 1, key="sky_layers")
        with c[3]: step      = st.slider("Layer step", 0.01, 0.10, 0.04, 0.005, key="sky_step")
        with c[4]: quality   = st.slider("Quality (samples)", 120, 800, 240, 10, key="sky_quality")
        with c[5]: seed      = st.number_input("Seed", value=19, step=1, key="sky_seed")

        v = st.columns(3)
        with v[0]: beam_n    = st.slider("Light beams", 0, 40, 10, 1, key="sky_beams")
        with v[1]: beam_h    = st.slider("Beam height", 0.10, 1.20, 0.55, 0.05, key="sky_beam_h")
        with v[2]: beam_skew = st.slider("Beam skew", -0.40, 0.40, 0.10, 0.01, key="sky_skew")

        np.random.seed(int(seed))
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_aspect('equal'); ax.axis('off')
        ax.set_xlim(-1.8, 1.8); ax.set_ylim(-0.1, 1.6)

        # Build a smooth skyline once
        xs = np.linspace(-1.6, 1.6, buildings)
        base = 0.3 + 0.5*np.random.rand(buildings)
        smooth = 0.33*np.roll(base,1) + 0.34*base + 0.33*np.roll(base,-1)
        heights = np.clip(smooth + rough*(np.random.rand(buildings)-0.5), 0.1, 1.2)
        sky_pts = np.column_stack([xs, heights])

        segs = catmull_rom_to_bezier(sky_pts, closed=False)
        sps = max(6, int(quality / max(1, len(segs))))
        top_curve = render_bezier_chain(segs, samples_per_seg=sps)

        # Ensure monotonic x for interp/fill_between
        order = np.argsort(top_curve[:,0])
        x_top = top_curve[order, 0]
        y_top = top_curve[order, 1]

        # Fast layered fill
        palette = plt.cm.viridis(np.linspace(0.25, 0.95, layers))
        for i in range(layers):
            y_layer = y_top - i*step
            y_layer = np.where(y_layer > 0.0, y_layer, 0.0)
            ax.fill_between(x_top, 0.0, y_layer, color=palette[i], alpha=0.12, linewidth=0.0, step=None)

        # Outline
        ax.plot(x_top, y_top, color="#1f2937", lw=2.0, alpha=0.9, zorder=layers+2)

        # Windows (vectorized)
        nwin = min(600, buildings*8)
        if nwin > 0:
            win_x = np.random.uniform(x_top.min(), x_top.max(), nwin)
            sky_y = np.interp(win_x, x_top, y_top)
            low = 0.05
            win_y = low + (sky_y - low) * np.random.rand(nwin)
            mask = win_y < sky_y
            ax.scatter(win_x[mask], win_y[mask], s=2, c="#ffd166", alpha=0.4, linewidths=0, zorder=layers+3)

        # Light beams (few, cheap)
        if beam_n > 0:
            idx = np.argsort(heights)[-beam_n:]
            bx = xs[idx]; by = heights[idx]
            for (x0, y0) in zip(bx, by):
                P0 = np.array([x0, y0])
                P1 = P0 + np.array([beam_skew*0.3, beam_h*0.4])
                P2 = P0 + np.array([beam_skew*0.7, beam_h*0.8])
                P3 = P0 + np.array([beam_skew,     beam_h])
                beam = cubic_bezier(P0, P1, P2, P3, N=120)
                ax.plot(beam[:,0], beam[:,1], color="#f8fafc", lw=2.0, alpha=0.18, zorder=layers+4)

        if st.button("💾 Save PNG (Topo City Skyline)", key="sky_save"):
            fname = f"gallery/bezier_skyline_{uuid.uuid4().hex[:6]}.png"
            save_current_figure(fig, fname); st.success(f"Saved: {fname}")

        st.pyplot(fig)

# --------------------
# Tab 6: Airfoil Designer (Presets + independent upper/lower Béziers)
# --------------------
with tab6:
    st.header("Airfoil Designer")

    mode = st.radio("Airfoil mode", ["Preset", "Custom Bézier"], horizontal=True, key="af_mode")
    N_plot = st.slider("Samples per surface (plot quality)", 100, 1200, 500, step=50, key="af_samples")

    if mode == "Preset":
        preset = st.selectbox(
            "Choose preset",
            [
                "NACA 0006", "NACA 0009", "NACA 0012", "NACA 0015",
                "NACA 2408", "NACA 2412", "NACA 2415", "NACA 2421",
                "NACA 4412", "NACA 4415", "NACA 4421",
                "NACA 63-412 (6-series, fetch)",
                "NACA 64-212 (6-series, fetch)",
                "NACA 65-410 (6-series, fetch)",
            ],
            index=2,
            key="af_preset"
        )

        st.caption("Tip: Type **any** NACA 4-digit code below (e.g., 2315, 4418) to generate instantly.")
        custom_4 = st.text_input("Custom NACA 4-digit code (optional)", key="af_custom4")

        up = st.file_uploader("…or upload airfoil coordinates (DAT/CSV with x,y columns)", type=["dat","txt","csv"], key="af_upload")

        # produce (xu,yu), (xl,yl)
        if up is not None:
            content = up.read()
            try:
                text = content.decode("utf-8", errors="ignore")
            except:
                text = str(content)
            rows = []
            for ln in text.splitlines():
                parts = re.split(r"[,\s]+", ln.strip())
                if len(parts) >= 2:
                    try:
                        rows.append([float(parts[0]), float(parts[1])])
                    except:
                        pass
            arr = np.array(rows, float)
            if len(arr) < 40:
                st.error("Could not parse enough numeric (x,y) rows from the upload."); st.stop()

            x = arr[:,0]; y = arr[:,1]
            xmin, xmax = x.min(), x.max()
            chord = max(xmax - xmin, 1e-9)
            xN = (x - xmin) / chord

            xg = np.linspace(0, 1, 501)
            yu = np.full_like(xg, np.nan, float)
            yl = np.full_like(xg, np.nan, float)
            for i, xv in enumerate(xg):
                mask = np.abs(xN - xv) < 0.01
                if not np.any(mask): continue
                ys = y[mask]
                yu[i] = np.nanmax(ys)
                yl[i] = np.nanmin(ys)
            def _fill(z):
                idx = ~np.isnan(z)
                return np.interp(xg, xg[idx], z[idx])
            (xu, yu), (xl, yl) = (xg, _fill(yu)), (xg, _fill(yl))

        else:
            if re.fullmatch(r"\d{4}", (custom_4 or "").strip()):
                (xu, yu), (xl, yl) = naca4_coords(custom_4.strip(), n=501)
            elif re.fullmatch(r"NACA\s+(\d{4})", preset):
                code4 = re.findall(r"(\d{4})", preset)[0]
                (xu, yu), (xl, yl) = naca4_coords(code4, n=501)
            else:
                series_name = preset.split(" (")[0]
                got = fetch_airfoil_coords_from_web(series_name)
                if got is None:
                    st.warning(f"Couldn’t fetch coordinates for **{series_name}**. Upload a DAT/CSV or pick a 4-digit preset.")
                    st.stop()
                (xu, yu), (xl, yl) = got

        # Fit cubic Bézier for display
        (P0u,P1u,P2u,P3u), (P0l,P1l,P2l,P3l) = fit_cubic_bezier_to_airfoil(xu, yu, xl, yl)
        upper = cubic_bezier(P0u, P1u, P2u, P3u, N_plot)
        lower = cubic_bezier(P0l, P1l, P2l, P3l, N_plot)

        x_grid = np.linspace(0, 1, 300)
        def interp_y(curve, xs):
            idx = np.abs(curve[:,0][:,None] - xs[None,:]).argmin(axis=0)
            return curve[idx,1]
        yu_i = interp_y(upper, x_grid)
        yl_i = interp_y(lower, x_grid)
        thickness = yu_i - yl_i
        camber = 0.5*(yu_i + yl_i)
        tmax = thickness.max(); xmax = x_grid[thickness.argmax()]

        title_name = (f"NACA {custom_4.strip()}" if (custom_4 or "").strip() else preset)
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].plot(upper[:,0], upper[:,1], 'r', lw=2, label="Upper (Bézier fit)")
        ax[0].plot(lower[:,0], lower[:,1], 'b', lw=2, label="Lower (Bézier fit)")
        ax[0].plot([0,1],[0,0],'k--', alpha=0.3, label="Chord")
        ax[0].set_aspect('equal','box'); ax[0].set_xlim(-0.05,1.05)
        pad = max(0.05, float(np.abs(np.concatenate([upper[:,1], lower[:,1]])).max()) + 0.05)
        ax[0].set_ylim(-pad, pad); ax[0].legend(); ax[0].set_title(title_name)

        ax[1].plot(x_grid, thickness, label="Thickness")
        ax[1].plot(x_grid, camber, label="Camber")
        ax[1].axvline(xmax, ls='--', alpha=0.4)
        ax[1].set_title(f"t_max={tmax:.3f} at x={xmax:.2f}")
        ax[1].set_xlabel("x / chord"); ax[1].legend()
        st.pyplot(fig)

        buf = io.StringIO(); w = csv.writer(buf)
        w.writerow(["x","y_upper","y_lower","thickness","camber"])
        for i, x in enumerate(x_grid):
            w.writerow([f"{x:.6f}", f"{yu_i[i]:.6f}", f"{yl_i[i]:.6f}", f"{thickness[i]:.6f}", f"{camber[i]:.6f}"])
        fname = (title_name.split(" (")[0]).replace(" ", "_")
        st.download_button("⬇️ Download preset (CSV)", buf.getvalue(),
                           file_name=f"{fname}.csv", mime="text/csv")

    else:
        st.subheader("Custom cubic Bézier surfaces (independent controls)")
        cols = st.columns(3)
        with cols[0]:
            te_thickness = st.slider("TE thickness (fraction of chord)", 0.0, 0.06, 0.00, step=0.005, key="af_te_t")
            indep_lower = st.checkbox("Independent lower surface controls", True, key="af_indep")
        with cols[1]:
            le_angle_u = st.slider("Upper LE tangent angle (deg from +x)", 0.0, 90.0, 20.0, step=1.0, key="af_lea_u")
            le_len_u   = st.slider("Upper LE handle length", 0.01, 0.9, 0.18, step=0.01, key="af_lel_u")
            te_angle_u = st.slider("Upper TE tangent angle (deg to +x)", -30.0, 30.0,  0.0, step=1.0, key="af_tea_u")
            te_len_u   = st.slider("Upper TE handle length", 0.01, 0.9, 0.25, step=0.01, key="af_tel_u")
        with cols[2]:
            if indep_lower:
                le_angle_l = st.slider("Lower LE tangent angle (deg from +x)", -90.0, 0.0, -20.0, step=1.0, key="af_lea_l")
                le_len_l   = st.slider("Lower LE handle length", 0.01, 0.9, 0.18, step=0.01, key="af_lel_l")
                te_angle_l = st.slider("Lower TE tangent angle (deg to +x)", -30.0, 30.0,  0.0, step=1.0, key="af_tea_l")
                te_len_l   = st.slider("Lower TE handle length", 0.01, 0.9, 0.25, step=0.01, key="af_tel_l")
            else:
                st.markdown("Lower surface mirrors the upper angles with sign flip on y; lengths matched.")

        LE = np.array([0.0, 0.0])
        TEu = np.array([1.0, +te_thickness/2.0])
        TEl = np.array([1.0, -te_thickness/2.0])

        le_dir_u = np.array([np.cos(np.radians(le_angle_u)), np.sin(np.radians(le_angle_u))])
        te_dir_u = np.array([np.cos(np.radians(te_angle_u)), np.sin(np.radians(te_angle_u))])

        if indep_lower:
            le_dir_l = np.array([np.cos(np.radians(le_angle_l)), np.sin(np.radians(le_angle_l))])
            te_dir_l = np.array([np.cos(np.radians(te_angle_l)), np.sin(np.radians(te_angle_l))])
        else:
            le_dir_l = np.array([le_dir_u[0], -le_dir_u[1]])
            te_dir_l = np.array([te_dir_u[0], -te_dir_u[1]])
            le_len_l, te_len_l = le_len_u, te_len_u

        P0u = LE
        P1u = LE + le_len_u * le_dir_u
        P2u = TEu - te_len_u * te_dir_u
        P3u = TEu

        P0l = LE
        P1l = LE + le_len_l * le_dir_l
        P2l = TEl - te_len_l * te_dir_l
        P3l = TEl

        upper = cubic_bezier(P0u, P1u, P2u, P3u, N_plot)
        lower = cubic_bezier(P0l, P1l, P2l, P3l, N_plot)

        x_grid = np.linspace(0, 1, 300)
        def interp_y(curve, xs):
            idx = np.abs(curve[:,0][:,None] - xs[None,:]).argmin(axis=0)
            return curve[idx,1]
        yu = interp_y(upper, x_grid); yl = interp_y(lower, x_grid)
        thickness = yu - yl; camber = 0.5*(yu + yl)
        tmax = thickness.max(); xmax = x_grid[thickness.argmax()]

        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].plot(upper[:,0], upper[:,1], 'r', lw=2, label="Upper")
        ax[0].plot(lower[:,0], lower[:,1], 'b', lw=2, label="Lower")
        if st.checkbox("Show control polygons", value=True, key="af_polys"):
            Cu = np.vstack([P0u, P1u, P2u, P3u])
            Cl = np.vstack([P0l, P1l, P2l, P3l])
            ax[0].plot(Cu[:,0], Cu[:,1], 'ro--', alpha=0.7)
            ax[0].plot(Cl[:,0], Cl[:,1], 'bo--', alpha=0.7)
        ax[0].plot([0,1],[0,0],'k--', alpha=0.3, label="Chord")
        ax[0].set_aspect('equal','box'); ax[0].set_xlim(-0.05,1.05)
        pad = max(0.05, float(np.abs(np.concatenate([upper[:,1], lower[:,1]])).max()) + 0.05)
        ax[0].set_ylim(-pad, pad); ax[0].legend()
        ax[0].set_title("Custom Bézier airfoil")

        ax[1].plot(x_grid, thickness, label="Thickness")
        ax[1].plot(x_grid, camber, label="Camber")
        ax[1].axvline(xmax, ls='--', alpha=0.4)
        ax[1].set_title(f"t_max={tmax:.3f} at x={xmax:.2f}")
        ax[1].set_xlabel("x / chord"); ax[1].legend()
        st.pyplot(fig)

        csv_buf = io.StringIO(); w = csv.writer(csv_buf)
        w.writerow(["x", "y_upper", "y_lower", "thickness", "camber"])
        for i, x in enumerate(x_grid):
            w.writerow([f"{x:.5f}", f"{yu[i]:.5f}", f"{yl[i]:.5f}", f"{thickness[i]:.5f}", f"{camber[i]:.5f}"])
        st.download_button("⬇️ Download airfoil (CSV)", csv_buf.getvalue(),
                           file_name=f"airfoil_custom_{uuid.uuid4().hex[:6]}.csv",
                           mime="text/csv")

# --------------------
# Tab 7: Path Planning 
# --------------------
with tab7:
    st.header("Path Planning")

    gcol = st.columns(4)
    with gcol[0]:
        x0 = st.number_input("Start x", value=-1.5, step=0.1, key="path_x0")
        y0 = st.number_input("Start y", value= 0.0, step=0.1, key="path_y0")
    with gcol[1]:
        x3 = st.number_input("Goal x",  value=+1.5, step=0.1, key="path_x3")
        y3 = st.number_input("Goal y",  value= 0.0, step=0.1, key="path_y3")
    with gcol[2]:
        samples = st.slider("Samples along path", 100, 1200, 600, step=50, key="path_samples")
    with gcol[3]:
        seed = st.number_input("Random seed", value=42, step=1, key="seed_path")

    st.subheader("Obstacles (circles)")
    n_obs = st.slider("Number of obstacles", 0, 12, 3, key="path_nobs")
    obstacles = []
    for i in range(n_obs):
        c1, c2, c3 = st.columns(3)
        with c1: ox = st.number_input(f"Obs {i+1} x", value=float(-0.5 + 0.5*i), step=0.1, key=f"path_ox{i}")
        with c2: oy = st.number_input(f"Obs {i+1} y", value=(0.6 if i%2==0 else -0.6), step=0.1, key=f"path_oy{i}")
        with c3: orad = st.number_input(f"Obs {i+1} r", value=0.35, min_value=0.01, step=0.05, key=f"path_or{i}")
        obstacles.append((ox, oy, orad))

    st.subheader("Objective Weights & Safety")
    wcol = st.columns(4)
    with wcol[0]: w_len = st.slider("Weight: length",    0.0, 5.0, 1.0, 0.1, key="w_len")
    with wcol[1]: w_cur = st.slider("Weight: curvature", 0.0, 5.0, 0.4, 0.1, key="w_cur")
    with wcol[2]: w_col = st.slider("Weight: collision", 1.0, 200.0, 40.0, 1.0, key="w_col")
    with wcol[3]: margin = st.slider("Clearance margin", 0.0, 0.8, 0.10, 0.01, key="w_margin")

    st.subheader("Optimizer")
    ccol = st.columns(5)
    with ccol[0]: iters = st.slider("Iterations", 10, 300, 80, 10, key="cem_iters")
    with ccol[1]: pop   = st.slider("Population", 50, 1000, 250, 50, key="cem_pop")
    with ccol[2]: elite_frac = st.slider("Elite fraction", 0.05, 0.5, 0.2, 0.05, key="cem_elite")
    with ccol[3]: init_spread = st.slider("Init spread (× chord)", 0.01, 1.0, 0.25, 0.01, key="cem_spread")
    with ccol[4]: cov_decay   = st.slider("Covariance decay", 0.80, 1.00, 0.95, 0.01, key="cem_decay")

    show_ctrl = st.checkbox("Show control polygon", True, key="show_ctrl_polygon")

    P0 = np.array([x0, y0]); P3 = np.array([x3, y3])
    chord = np.linalg.norm(P3 - P0) + 1e-9
    n = (P3 - P0) / chord
    n_perp = np.array([-n[1], n[0]])
    P1_init = P0 + (1/3.0)*(P3-P0) + 0.2*chord*n_perp
    P2_init = P0 + (2/3.0)*(P3-P0) - 0.2*chord*n_perp

    def cost(P1, P2):
        pts = cubic_bezier(P0, P1, P2, P3, samples)
        J_len = poly_length(pts)
        kappa = numeric_curvature(pts)
        J_cur = np.mean(kappa**2)
        J_col = 0.0
        if obstacles:
            px, py = pts[:,0], pts[:,1]
            for (ox, oy, r) in obstacles:
                d = np.sqrt((px-ox)**2 + (py-oy)**2)
                pen = np.maximum(0.0, (r + margin) - d)
                J_col += np.mean(pen*pen)
            J_col *= 100.0
        return w_len*J_len + w_cur*J_cur + w_col*J_col, (J_len, J_cur, J_col)

    def cem_optimize(P1_0, P2_0, iters, pop, elite_frac, init_spread, cov_decay, seed=0):
        rng = np.random.default_rng(int(seed))
        mu = np.hstack([P1_0, P2_0])
        sigma = (init_spread*chord) * np.ones(4)
        elite = max(2, int(pop*elite_frac))
        best_vec = mu.copy()
        best_cost = float('inf')
        best_breakdown = (0,0,0)

        for _ in range(iters):
            samples_vec = rng.normal(mu, sigma, size=(pop, 4))
            costs = []; brk = []
            for v in samples_vec:
                P1 = v[:2]; P2 = v[2:]
                J, breakdown = cost(P1, P2)
                costs.append(J); brk.append(breakdown)
            idx = np.argsort(costs)
            elites = samples_vec[idx[:elite]]
            mu = elites.mean(axis=0)
            sigma = np.maximum(1e-4, elites.std(axis=0)) * cov_decay
            if costs[idx[0]] < best_cost:
                best_cost = costs[idx[0]]
                best_vec = samples_vec[idx[0]]
                best_breakdown = brk[idx[0]]

        return best_vec[:2], best_vec[2:], best_cost, best_breakdown

    run = st.button("🚀 Optimize", key="run_optimize")
    if run:
        P1_opt, P2_opt, Jopt, (Jlen, Jcur, Jcol) = cem_optimize(
            P1_init, P2_init, iters, pop, elite_frac, init_spread, cov_decay, seed
        )
    else:
        P1_opt, P2_opt = P1_init, P2_init
        Jopt, (Jlen, Jcur, Jcol) = cost(P1_opt, P2_opt)

    path = cubic_bezier(P0, P1_opt, P2_opt, P3, samples)

    fig, ax = plt.subplots(figsize=(8, 6))
    for (ox, oy, r) in obstacles:
        circ = plt.Circle((ox, oy), r, color='tab:red', alpha=0.25, ec='k', lw=1)
        ax.add_patch(circ)
        if margin > 0:
            saf = plt.Circle((ox, oy), r+margin, color='tab:red', alpha=0.08, ec='r', lw=1, ls='--')
            ax.add_patch(saf)

    ax.plot(path[:,0], path[:,1], lw=3, label="Optimal Bézier path")
    if show_ctrl:
        ctrl = np.vstack([P0, P1_opt, P2_opt, P3])
        ax.plot(ctrl[:,0], ctrl[:,1], 'o--', label="Control polygon")

    ax.plot([P0[0]], [P0[1]], 'go', ms=9, label="Start")
    ax.plot([P3[0]], [P3[1]], 'ro', ms=9, label="Goal")

    xs = [P0[0], P3[0]] + [ox for (ox,_,_) in obstacles]
    ys = [P0[1], P3[1]] + [oy for (_,oy,_) in obstacles]
    rs = [0,0] + [r for (_,_,r) in obstacles]
    if xs:
        pad = 0.5 + 0.3*(len(obstacles)>0)
        xmin = min([x - rr - pad for x, rr in zip(xs, rs)])
        xmax = max([x + rr + pad for x, rr in zip(xs, rs)])
        ymin = min([y - rr - pad for y, rr in zip(ys, rs)])
        ymax = max([y + rr + pad for y, rr in zip(ys, rs)])
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)

    ax.set_aspect('equal', 'box')
    ax.set_title("Optimal Path (Cubic Bézier)")
    ax.legend(loc="upper right")

    st.pyplot(fig)
    colm = st.columns(4)
    with colm[0]: st.metric("Total cost J", f"{Jopt:.4f}")
    with colm[1]: st.metric("Length term", f"{Jlen:.4f}")
    with colm[2]: st.metric("Curvature term", f"{Jcur:.4f}")
    with colm[3]: st.metric("Collision term", f"{Jcol:.4f}")

    buf = io.StringIO(); w = csv.writer(buf)
    w.writerow(["x","y"]); [w.writerow([f"{x:.6f}", f"{y:.6f}"]) for x,y in path]
    st.download_button("⬇️ Download path (CSV)", buf.getvalue(),
                       file_name=f"bezier_opt_path_{uuid.uuid4().hex[:6]}.csv", mime="text/csv")

    buf2 = io.StringIO(); w2 = csv.writer(buf2)
    w2.writerow(["ControlPoint","x","y"])
    for name, p in zip(["P0","P1","P2","P3"], [P0,P1_opt,P2_opt,P3]):
        w2.writerow([name, f"{p[0]:.6f}", f"{p[1]:.6f}"])
    st.download_button("⬇️ Download control points (CSV)", buf2.getvalue(),
                       file_name=f"bezier_controls_{uuid.uuid4().hex[:6]}.csv", mime="text/csv")

    st.caption("Objective = w_len·length + w_cur·mean(κ²) + w_col·⟨ReLU(r+margin−d)²⟩·100. "
               "Increase collision weight/margin for safer paths; increase curvature weight for smoother paths; "
               "raise population/iterations for higher-quality solutions.")
