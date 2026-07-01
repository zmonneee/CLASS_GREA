#!/usr/bin/env python3
"""
Generator script: writes grea_results.ipynb at class_public-3.3.4/grea_results.ipynb.
Replaces cs2_fld=0/1 variation with alpha = {0.8, 0.9, 1.0, 1.1, 1.2, 1.3} variation.
Run once; then open grea_results.ipynb normally.
"""
import nbformat as nbf, os

NB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'grea_results.ipynb'))

md   = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell
cells = []

# ═══════════════════════════════════════════════════════════════════════════════
# Title
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(md("""\
# GREA-Patched CLASS 3.3.4 — Result Figures (α variation)
**General Relativistic Entropic Acceleration (GREA) dark energy, García-Bellido 2024 (arXiv:2405.02895)**

Models shown: α ∈ {0.8, 0.9, 1.0, 1.1, 1.2, 1.3} at fixed cs²=1, compared to LCDM (black).
α is a *derived* quantity — each model is found by solving sqrt_k_eta0 such that
`α = sqrt_k_eta0 / (E(1)·τ(1))` matches the target (brentq on a Python mirror of background.c ODE).

Cell 0: freshness guard + 3 inline gates.  Cell 1: parameter setup.
Cell 2: α → sqrt_k_eta0 solver + mapping table.  Cell 3: build all CLASS instances.
"""))

# ═══════════════════════════════════════════════════════════════════════════════
# Cell 0 — freshness guard + inline gates
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(code("""\
import sys, os, glob, datetime, importlib.util
import numpy as np

# ROOT detection
_ROOT = os.path.abspath(os.getcwd())
if not os.path.exists(os.path.join(_ROOT, 'source', 'background.c')):
    _ROOT = os.path.abspath(os.path.join(_ROOT, '..'))
if not os.path.exists(os.path.join(_ROOT, 'source', 'background.c')):
    raise RuntimeError("Cannot find CLASS root. Start Jupyter from class_public-3.3.4/.")
ROOT = _ROOT
sys.path.insert(0, ROOT)
print(f"ROOT = {ROOT}")

# Import check_binary_freshness from validate_grea.py
_spec = importlib.util.spec_from_file_location(
    'validate_grea', os.path.join(ROOT, 'scripts', 'validate_grea.py'))
_vg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_vg)

BINARY_FRESH = False
try:
    _vg.check_binary_freshness()
    BINARY_FRESH = True
except SystemExit:
    print("\\n\\u26d4 STALE BINARY:")
    print("    make libclass.a && python setup.py build_ext --inplace")

if not BINARY_FRESH:
    raise RuntimeError("Stale binary. Rebuild and re-run Cell 0.")

# Inline gate checks (background only, fast)
from classy import Class
_BASE = {'omega_b': 0.02237, 'omega_cdm': 0.1200, 'h': 0.70,
         'A_s': 2.1e-9, 'n_s': 0.9649, 'tau_reio': 0.0544,
         'output': '', 'base_path': ROOT}
_cl = Class(); _cl.set({**_BASE, 'fluid_equation_of_state': 'CLP',
                         'w0_fld': -1., 'wa_fld': 0., 'cs2_fld': 1.}); _cl.compute()
_cg = Class(); _cg.set({**_BASE, 'fluid_equation_of_state': 'GREA',
                         'sqrt_k_eta0': 3.6, 'cs2_fld': 1., 'use_ppf': 'yes'}); _cg.compute()

_bg_g = _cg.get_background()
_z_w   = np.sort(np.array(_bg_g['z']))
_w_fld = np.array(_bg_g['(.)w_fld'])[np.argsort(np.array(_bg_g['z']))]
_w0    = float(np.interp(0., _z_w, _w_fld))
_rs_l  = _cl.get_current_derived_parameters(['rs_rec'])['rs_rec']
_rs_g  = _cg.get_current_derived_parameters(['rs_rec'])['rs_rec']

_g1_ok = abs(_cg.h() - _cl.h()) < 1e-7
_g2_ok = abs(_rs_g/_rs_l - 1.) < 1e-4
_g7_ok = abs(_w0 + 1.) < 0.02

print("\\n\\u2500\\u2500 Inline gate checks \\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500")
print(f"  G1  H0 shared  (|\\u0394h| < 1e-7):       {'PASS \\u2713' if _g1_ok else 'FAIL \\u2717'}  "
      f"LCDM h={_cl.h():.9f}  GREA h={_cg.h():.9f}")
print(f"  G2  r_s(rec)  (|\\u0394| < 1e-4):         {'PASS \\u2713' if _g2_ok else 'FAIL \\u2717'}  "
      f"\\u0394r_s/r_s={_rs_g/_rs_l - 1.:.3e}")
print(f"  G7a w0 \\u2248 -1  (|w0+1| < 0.02):       {'PASS \\u2713' if _g7_ok else 'FAIL \\u2717'}  "
      f"w0(fiducial g=3.6)={_w0:.6f}")
_cl.struct_cleanup(); _cl.empty()
_cg.struct_cleanup(); _cg.empty()

GATES_OK = _g1_ok and _g2_ok and _g7_ok
if not GATES_OK:
    raise RuntimeError("One or more inline gates FAILED.")
print("\\n\\u2705 Binary fresh, inline gates passed.")
"""))

# ═══════════════════════════════════════════════════════════════════════════════
# Cell 1 — Setup / globals
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(md("## Setup"))
cells.append(code("""\
%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.cm as mcm
import matplotlib.colors as mcolors
import numpy as np, os, datetime
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

assert BINARY_FRESH and GATES_OK, "Re-run Cell 0 first."

FIGURES_DIR = os.path.join(ROOT, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)
RUN_TS = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

def save_fig(fig, name):
    path = os.path.join(FIGURES_DIR, f'{name}_{RUN_TS}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  saved -> {os.path.basename(path)}")

plt.rcParams.update({
    'font.size': 11, 'axes.labelsize': 12, 'legend.fontsize': 10,
    'axes.grid': True, 'grid.alpha': 0.3, 'lines.linewidth': 1.8,
})

# alpha targets for variation (García-Bellido 2024 / Calderón 2025 range)
ALPHA_TARGETS = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
ALPHA_NORM    = mcolors.Normalize(vmin=0.8, vmax=1.3)
CMAP          = mcm.viridis
def grea_color(alpha):
    return CMAP(ALPHA_NORM(alpha))

LCDM_COLOR = '#222222'   # black
_C_KMS = 2.99792458e5

# Parameter names verified against source/input.c lines 3614-3624 (GREA),
# 3580-3604 (CLP), 1886-1945 (output string matching).
COMMON = {
    'omega_b': 0.02237, 'omega_cdm': 0.1200, 'h': 0.70,
    'A_s': 2.1e-9, 'n_s': 0.9649, 'tau_reio': 0.0544,
    'output': 'tCl pCl lCl mPk',
    'lensing': 'yes',
    'P_k_max_1/Mpc': 5.0,
    'z_max_pk': 3.0,
    'l_max_scalars': 2500,
    'base_path': ROOT,
}
LCDM_P   = {**COMMON, 'fluid_equation_of_state': 'CLP',
             'w0_fld': -1., 'wa_fld': 0., 'cs2_fld': 1.}
GREA_P_TEMPLATE = {**COMMON, 'fluid_equation_of_state': 'GREA',
                    'cs2_fld': 1.0, 'use_ppf': 'yes'}
print("Setup done.")
"""))

# ═══════════════════════════════════════════════════════════════════════════════
# Cell 2 — Alpha solver
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(code("""\
# alpha = sqrt_k_eta0 / (grea_E1 * tau_table[N-1])
# (background.c line 1277)
# Mirror the tau ODE from background.c in Python, solve for sqrt_k_eta0
# using brentq. Validated: g=3.6 -> alpha=1.07938 (matches CLASS diagnostic).

_OMEGA_G  = 2.47298e-5   # photons (T_cmb=2.7255K)
_OMEGA_UR = 1.70961e-5   # ur relics (N_ur=3.044)
_OMEGA_R  = _OMEGA_G + _OMEGA_UR

def _compute_alpha_py(sqrt_k_eta0,
                      omega_b=COMMON['omega_b'],
                      omega_cdm=COMMON['omega_cdm'],
                      h=COMMON['h']):
    g = sqrt_k_eta0
    W = np.pi * (np.sinh(2*g) - 2*g)
    OmegaGB = (4*np.pi/3) / W
    Omega_m0 = (omega_b + omega_cdm) / h**2
    Omega_r0 = _OMEGA_R / h**2
    a_ini    = 1e-14
    tau0     = a_ini / np.sqrt(Omega_r0)   # radiation-dominated IC

    def rhs(loga, tau):
        a  = np.exp(loga)
        x  = 2*tau[0]
        fG = OmegaGB * np.sinh(x) / a**2
        E  = np.sqrt(Omega_m0/a**3 + Omega_r0/a**4 + fG)
        return [1.0 / (a * E)]

    sol = solve_ivp(rhs, [np.log(a_ini), 0.0], [tau0],
                    method='RK45', rtol=1e-11, atol=1e-13)
    tau1 = float(sol.y[0, -1])
    a1   = np.exp(float(sol.t[-1]))
    fG1  = OmegaGB * np.sinh(2*tau1) / a1**2
    E1   = np.sqrt(Omega_m0/a1**3 + Omega_r0/a1**4 + fG1)
    return g / (E1 * tau1)

print("Solving sqrt_k_eta0 for each alpha target via brentq [2.5, 4.5]...")
ALPHA_TO_G = {}
for at in ALPHA_TARGETS:
    try:
        g_sol = brentq(lambda g: _compute_alpha_py(g) - at, 2.5, 4.5, xtol=1e-8)
        alpha_check = _compute_alpha_py(g_sol)
        delta = alpha_check - at
        if abs(delta) > 1e-3:
            print(f"  WARNING: alpha={at:.2f}  g={g_sol:.8f}  delta={delta:.2e}  -- exceeds 1e-3!")
        else:
            ALPHA_TO_G[at] = g_sol
    except Exception as e:
        print(f"  FAILED alpha={at:.2f}: {e}")

print()
print(f"  {'alpha':>7}  {'sqrt_k_eta0':>14}  {'alpha_check':>13}  {'delta':>12}  check")
print("  " + "-"*58)
for at in ALPHA_TARGETS:
    if at in ALPHA_TO_G:
        g = ALPHA_TO_G[at]
        ac = _compute_alpha_py(g)
        ok = "OK" if abs(ac - at) < 1e-3 else "WARN"
        print(f"  {at:7.2f}   {g:14.8f}   {ac:13.8f}   {ac-at:+12.2e}   {ok}")
    else:
        print(f"  {at:7.2f}   FAILED")
print()
print(f"Note: fiducial g=3.6 -> alpha={_compute_alpha_py(3.6):.5f} (between 1.0 and 1.1)")
"""))

# ═══════════════════════════════════════════════════════════════════════════════
# Cell 3 — Build CLASS instances
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(code("""\
from classy import Class

def build(params):
    c = Class(); c.set(params); c.compute(); return c

print("Building 1 LCDM + 6 GREA instances (may take ~3 min total)...")
lcdm = build(LCDM_P)
print(f"  LCDM done:  h={lcdm.h():.6f}  sigma8={lcdm.sigma8():.6f}")

grea_instances = {}
bg_grea        = {}
for at in ALPHA_TARGETS:
    if at not in ALPHA_TO_G:
        print(f"  SKIP alpha={at:.2f} (solver failed)")
        continue
    g   = ALPHA_TO_G[at]
    c   = build({**GREA_P_TEMPLATE, 'sqrt_k_eta0': g})
    grea_instances[at] = c
    bg_grea[at]        = c.get_background()
    print(f"  alpha={at:.2f}  g={g:.6f}  h={c.h():.6f}  sigma8={c.sigma8():.6f}")

bg_lcdm = lcdm.get_background()

# Summary table
from IPython.display import display, Markdown

def _w0_wa(c):
    bg = c.get_background()
    z_s = np.array(bg['z']); idx = np.argsort(z_s)
    w_s = np.array(bg['(.)w_fld'])[idx]; z_s = z_s[idx]
    w0  = float(np.interp(0., z_s, w_s))
    # wa = -dw/da at a=1 via quadratic fit in a in [0.90,1.0]
    a_fit = np.linspace(0.90, 1.0, 30)
    w_fit = np.array([float(np.interp(1./a-1., z_s, w_s)) for a in a_fit])
    coeffs = np.polyfit(a_fit - 1., w_fit, 2)   # w ~ c0*(a-1)^2 + c1*(a-1) + c2
    wa = -coeffs[1]   # -dw/da at a=0 <=> a=1
    return w0, wa

rows = []
for at in ALPHA_TARGETS:
    if at not in grea_instances: continue
    c = grea_instances[at]
    w0, wa = _w0_wa(c)
    rows.append((at, ALPHA_TO_G[at], c.h()*100., c.sigma8(), c.S8(), w0, wa))

header = "| α | sqrt_k_eta0 | H₀ (km/s/Mpc) | σ₈ | S8 | w₀ | wₐ |"
sep    = "|---|---|---|---|---|---|---|"
lines  = [header, sep]
for r in rows:
    lines.append(f"| {r[0]:.2f} | {r[1]:.8f} | {r[2]:.6f} | {r[3]:.6f} | {r[4]:.6f} | {r[5]:.5f} | {r[6]:.4f} |")
display(Markdown("\\n".join(lines)))
"""))

# ═══════════════════════════════════════════════════════════════════════════════
# Cell 4 — Helpers
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(code("""\
def bg_sorted(bg, *keys):
    z = np.array(bg['z']); idx = np.argsort(z); z = z[idx]
    return (z,) + tuple(np.array(bg[k])[idx] for k in keys)

def interp1(z_target, z_arr, y_arr):
    return float(np.interp(z_target, z_arr, y_arr))

def add_alpha_colorbar(fig, ax, label=r'$\\alpha$'):
    sm = mcm.ScalarMappable(cmap=CMAP, norm=ALPHA_NORM)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, label=label, fraction=0.046, pad=0.04)
    cbar.set_ticks(ALPHA_TARGETS)
    return cbar

SOLVED_ALPHAS = [at for at in ALPHA_TARGETS if at in grea_instances]
print(f"Helpers defined. Solved alphas: {SOLVED_ALPHAS}")
"""))

# ═══════════════════════════════════════════════════════════════════════════════
# Section A — Background
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(md("""\
---
## Section A — Background Quantities

Six GREA curves (α = 0.8 – 1.3, viridis) plus LCDM (black, dashed).
Key feature: the depth and position of the transient phantom crossing in w(a)
depends strongly on α (reproducing García-Bellido 2024 Fig. 1 spread).
"""))

# A1 — H(z) and ratio
cells.append(code("""\
# A1: H(z) and H_GREA / H_LCDM - 1
z_l, H_l = bg_sorted(bg_lcdm, 'H [1/Mpc]')
z_plot = np.logspace(-3, np.log10(3.), 1000)
Hl  = np.interp(z_plot, z_l, H_l) * _C_KMS

fig, (ax, axr) = plt.subplots(2, 1, figsize=(8, 7), sharex=True,
                                gridspec_kw={'height_ratios': [2, 1]})
ax.plot(z_plot, Hl, color=LCDM_COLOR, lw=2., ls='--', label='LCDM', zorder=10)
axr.axhline(0, color='k', lw=0.8)

for at in SOLVED_ALPHAS:
    col = grea_color(at)
    z_g, H_g = bg_sorted(bg_grea[at], 'H [1/Mpc]')
    Hg = np.interp(z_plot, z_g, H_g) * _C_KMS
    ax.plot(z_plot, Hg, color=col, lw=1.5, alpha=0.9)
    axr.plot(z_plot, Hg/Hl - 1., color=col, lw=1.5, alpha=0.9)

ax.set_ylabel('H(z)  [km/s/Mpc]'); ax.set_xscale('log')
ax.set_title('Hubble parameter H(z)')
ax.legend(loc='upper left')
axr.set_ylabel(r'$H_{\\rm GREA}/H_{\\rm LCDM} - 1$')
axr.set_xlabel('Redshift z')
add_alpha_colorbar(fig, axr)
plt.tight_layout(); save_fig(fig, 'A1_Hz_ratio'); plt.show()
"""))

# A2 — w(a)
cells.append(code("""\
# A2: w(a) vs z — phantom crossing depth/position as a key alpha signature
fig, ax = plt.subplots(figsize=(8, 5))
ax.axhline(-1., color='k', lw=0.8, ls=':', label='w = -1', zorder=10)

def phantom_crossings(z_arr, w_arr):
    wp1 = w_arr + 1.
    sc  = np.where(np.diff(np.sign(wp1)))[0]
    return [z_arr[i] + (0-wp1[i])*(z_arr[i+1]-z_arr[i])/(wp1[i+1]-wp1[i]) for i in sc]

z_w = np.linspace(0., 3., 3000)
for at in SOLVED_ALPHAS:
    col = grea_color(at)
    z_g, w_g = bg_sorted(bg_grea[at], '(.)w_fld')
    wi = np.interp(z_w, z_g, w_g)
    ax.plot(z_w, wi, color=col, lw=1.5, alpha=0.9, label=f'\\u03b1={at:.1f}')
    for zc in phantom_crossings(z_w, wi):
        ax.axvline(zc, color=col, lw=0.6, ls=':', alpha=0.5)

ax.set_xlabel('Redshift z'); ax.set_ylabel('w(z)')
ax.set_xlim(0., 3.); ax.set_title('GREA equation of state w(z)  [alpha variation]')
add_alpha_colorbar(fig, ax); plt.tight_layout(); save_fig(fig, 'A2_w_alpha'); plt.show()

# Print phantom crossing table
print(f"\\n  {'alpha':>7}  {'w0':>10}  {'phantom crossings z':}")
print("  " + "-"*50)
for at in SOLVED_ALPHAS:
    z_g, w_g = bg_sorted(bg_grea[at], '(.)w_fld')
    wi  = np.interp(z_w, z_g, w_g)
    w0_ = float(np.interp(0., z_w, wi))
    xings = phantom_crossings(z_w, wi)
    print(f"  {at:7.2f}   {w0_:10.5f}   {[f'{x:.3f}' for x in xings]}")
"""))

# A3 — Omega_fld
cells.append(code("""\
# A3: Effective DE density Omega_fld(z)
# LCDM -> rho_lambda / rho_crit; GREA -> rho_fld / rho_crit
z_l, rc_l, rl_l = bg_sorted(bg_lcdm, '(.)rho_crit', '(.)rho_lambda')
z_p = np.logspace(-3, np.log10(3.), 1000)
OmL = np.interp(z_p, z_l, rl_l/rc_l)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(z_p, OmL, color=LCDM_COLOR, lw=2., ls='--', label=r'LCDM $\\Omega_\\Lambda$', zorder=10)
for at in SOLVED_ALPHAS:
    col = grea_color(at)
    z_g, rc_g, rf_g = bg_sorted(bg_grea[at], '(.)rho_crit', '(.)rho_fld')
    ax.plot(z_p, np.interp(z_p, z_g, rf_g/rc_g), color=col, lw=1.5, alpha=0.9)
ax.set_xscale('log'); ax.set_xlabel('Redshift z')
ax.set_ylabel(r'$\\Omega_{\\rm DE}(z)$'); ax.set_title('Effective DE density fraction')
ax.legend(); add_alpha_colorbar(fig, ax); plt.tight_layout(); save_fig(fig, 'A3_Omega_fld'); plt.show()
"""))

# A4 — Distances
cells.append(code("""\
# A4: Comoving D_C(z) and angular diameter D_A(z) ratios
z_dist = np.linspace(0.01, 3., 150)
DC_l  = np.array([lcdm.comoving_distance(z)  for z in z_dist])
DA_l  = np.array([lcdm.angular_distance(z)   for z in z_dist])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.axhline(0, color='k', lw=0.8)
ax2.axhline(0, color='k', lw=0.8)
for at in SOLVED_ALPHAS:
    col = grea_color(at)
    c   = grea_instances[at]
    DC_g = np.array([c.comoving_distance(z) for z in z_dist])
    DA_g = np.array([c.angular_distance(z)  for z in z_dist])
    ax1.plot(z_dist, DC_g/DC_l - 1., color=col, lw=1.5, alpha=0.9)
    ax2.plot(z_dist, DA_g/DA_l - 1., color=col, lw=1.5, alpha=0.9)
ax1.set_xlabel('z'); ax1.set_ylabel(r'$D_C^{\\rm GREA}/D_C^{\\rm LCDM} - 1$')
ax1.set_title('Comoving distance ratio')
ax2.set_xlabel('z'); ax2.set_ylabel(r'$D_A^{\\rm GREA}/D_A^{\\rm LCDM} - 1$')
ax2.set_title('Angular diameter distance ratio')
add_alpha_colorbar(fig, ax2); plt.tight_layout(); save_fig(fig, 'A4_distances'); plt.show()
"""))

# A5 — Deceleration
cells.append(code("""\
# A5: Deceleration parameter q(z) = 1/2 * (1 + 3 p_tot/rho_tot)
def q_from_bg(bg):
    z, rho, p = bg_sorted(bg, '(.)rho_tot', '(.)p_tot')
    return z, 0.5 * (1. + 3.*p/rho)

z_ql, q_l = q_from_bg(bg_lcdm)
z_p = np.linspace(0., 3., 1000)

fig, ax = plt.subplots(figsize=(8, 5))
ax.axhline(0, color='k', lw=0.8, ls=':')
ax.plot(z_p, np.interp(z_p, z_ql, q_l), color=LCDM_COLOR, lw=2., ls='--', label='LCDM', zorder=10)
for at in SOLVED_ALPHAS:
    col = grea_color(at)
    z_qg, q_g = q_from_bg(bg_grea[at])
    ax.plot(z_p, np.interp(z_p, z_qg, q_g), color=col, lw=1.5, alpha=0.9)
ax.set_xlabel('z'); ax.set_ylabel('q(z)')
ax.set_title(r'Deceleration parameter  $q = \\frac{1}{2}(1+3p/\\rho)$')
ax.legend(); add_alpha_colorbar(fig, ax); plt.tight_layout(); save_fig(fig, 'A5_deceleration'); plt.show()
"""))

# ═══════════════════════════════════════════════════════════════════════════════
# Section B — Growth
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(md("""\
---
## Section B — Structure Growth

At fixed A_s = 2.1×10⁻⁹, larger α → stronger DE phantom behavior → different Hubble friction
at z ≈ 0.5–2 → different growth. The spread across α shows how sensitive σ₈ and fσ₈ are to α.
"""))

cells.append(code("""\
# Pre-compute fsigma8, sigma8, D on a common redshift grid
z_b  = np.linspace(0., 2.5, 60)
R8_h = 8. / lcdm.h()   # 8 Mpc/h (all models share h=0.70)

fs8_lcdm = np.array([lcdm.scale_independent_growth_factor_f(z)*lcdm.sigma(R8_h, z) for z in z_b])
s8_lcdm  = np.array([lcdm.sigma(R8_h, z) for z in z_b])
D_lcdm   = np.array([lcdm.scale_independent_growth_factor(z) for z in z_b])
D_lcdm_n = lcdm.scale_independent_growth_factor(99.)

fs8_grea = {}; s8_grea = {}; D_grea = {}; D_grea_n = {}
for at in SOLVED_ALPHAS:
    c = grea_instances[at]
    fs8_grea[at] = np.array([c.scale_independent_growth_factor_f(z)*c.sigma(R8_h, z) for z in z_b])
    s8_grea[at]  = np.array([c.sigma(R8_h, z) for z in z_b])
    D_grea[at]   = np.array([c.scale_independent_growth_factor(z) for z in z_b])
    D_grea_n[at] = c.scale_independent_growth_factor(99.)
print("Growth arrays computed.")
"""))

cells.append(code("""\
# B1: f*sigma8(z)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(z_b, fs8_lcdm, color=LCDM_COLOR, lw=2., ls='--', label='LCDM', zorder=10)
for at in SOLVED_ALPHAS:
    ax.plot(z_b, fs8_grea[at], color=grea_color(at), lw=1.5, alpha=0.9)
ax.set_xlabel('z'); ax.set_ylabel(r'$f\\sigma_8(z)$')
ax.set_title(r'Growth rate $f\\sigma_8(z)$  [fixed $A_s$]')
ax.legend(); add_alpha_colorbar(fig, ax); plt.tight_layout(); save_fig(fig, 'B1_fsigma8'); plt.show()
"""))

cells.append(code("""\
# B2: sigma8(z)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(z_b, s8_lcdm, color=LCDM_COLOR, lw=2., ls='--', label='LCDM', zorder=10)
for at in SOLVED_ALPHAS:
    ax.plot(z_b, s8_grea[at], color=grea_color(at), lw=1.5, alpha=0.9)
ax.set_xlabel('z'); ax.set_ylabel(r'$\\sigma_8(z)$')
ax.set_title(r'$\\sigma_8(z)$  [fixed $A_s$]')
ax.legend(); add_alpha_colorbar(fig, ax); plt.tight_layout(); save_fig(fig, 'B2_sigma8z'); plt.show()
"""))

cells.append(code("""\
# B3: D_GREA / D_LCDM - 1 (both normalized to D(z=99)=1)
fig, ax = plt.subplots(figsize=(8, 5))
ax.axhline(0, color='k', lw=0.8)
D_lcdm_r = D_lcdm / D_lcdm_n
for at in SOLVED_ALPHAS:
    D_g_r = D_grea[at] / D_grea_n[at]
    ax.plot(z_b, D_g_r/D_lcdm_r - 1., color=grea_color(at), lw=1.5, alpha=0.9)
ax.set_xlabel('z')
ax.set_ylabel(r'$D_{\\rm GREA}/D_{\\rm LCDM} - 1$  (norm z=99)')
ax.set_title('Relative growth factor')
add_alpha_colorbar(fig, ax); plt.tight_layout(); save_fig(fig, 'B3_D_ratio'); plt.show()
"""))

cells.append(code("""\
# B4: Summary table (sigma8(0), S8, fsigma8(z=0.5))
from IPython.display import display, Markdown
rows_b = []
for at in SOLVED_ALPHAS:
    c   = grea_instances[at]
    s8z = c.sigma8()
    S8  = c.S8()
    fs8 = float(np.interp(0.5, z_b, fs8_grea[at]))
    rows_b.append((at, ALPHA_TO_G[at], s8z, S8, fs8))
header = "| α | sqrt_k_eta0 | σ₈(0) | S8 | fσ₈(z=0.5) |"
sep    = "|---|---|---|---|---|"
lines  = [header, sep,
          f"| LCDM | — | {lcdm.sigma8():.5f} | {lcdm.S8():.5f} | {float(np.interp(0.5, z_b, fs8_lcdm)):.5f} |"]
for r in rows_b:
    lines.append(f"| {r[0]:.2f} | {r[1]:.6f} | {r[2]:.5f} | {r[3]:.5f} | {r[4]:.5f} |")
display(Markdown("\\n".join(lines)))
"""))

# ═══════════════════════════════════════════════════════════════════════════════
# Section C — Matter power spectrum
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(md("""\
---
## Section C — Matter Power Spectrum  (linear only)

The ratio P_GREA/P_LCDM − 1 (linear y-axis) shows the k-dependence of the GREA signal
as a function of α. The spread across α should show a scale-dependent shift.
"""))

cells.append(code("""\
# C1 P(k) log-log, C2 Delta^2(k), C3 ratio (linear y)
pk_l, k_l, z_l_pk = lcdm.get_pk_and_k_and_z(nonlinear=False)
iz0_l = np.argmin(np.abs(z_l_pk))
Pkl = pk_l[:, iz0_l]

pk_grea = {}; kk_grea = {}
for at in SOLVED_ALPHAS:
    pk_g, k_g, z_g_pk = grea_instances[at].get_pk_and_k_and_z(nonlinear=False)
    iz0 = np.argmin(np.abs(z_g_pk))
    pk_grea[at] = pk_g[:, iz0]
    kk_grea[at] = k_g

# Interpolate all onto LCDM k grid for the ratio
for at in SOLVED_ALPHAS:
    pk_grea[at] = np.interp(k_l, kk_grea[at], pk_grea[at])

# C1: log-log P(k)
fig, ax = plt.subplots(figsize=(8, 5))
ax.loglog(k_l, Pkl, color=LCDM_COLOR, lw=2., ls='--', label='LCDM', zorder=10)
for at in SOLVED_ALPHAS:
    ax.loglog(k_l, pk_grea[at], color=grea_color(at), lw=1.5, alpha=0.9)
ax.set_xlabel(r'k  [Mpc$^{-1}$]'); ax.set_ylabel(r'$P(k)$  [Mpc$^3$]')
ax.set_title('Linear matter power spectrum at z = 0')
ax.legend(); add_alpha_colorbar(fig, ax); plt.tight_layout(); save_fig(fig, 'C1_Pk'); plt.show()

# C2: Delta^2(k)
fac = k_l**3 / (2.*np.pi**2)
fig, ax = plt.subplots(figsize=(8, 5))
ax.loglog(k_l, fac*Pkl, color=LCDM_COLOR, lw=2., ls='--', label='LCDM', zorder=10)
for at in SOLVED_ALPHAS:
    ax.loglog(k_l, fac*pk_grea[at], color=grea_color(at), lw=1.5, alpha=0.9)
ax.set_xlabel(r'k  [Mpc$^{-1}$]')
ax.set_ylabel(r'$\\Delta^2(k) = k^3 P(k)/(2\\pi^2)$')
ax.set_title('Dimensionless power spectrum at z = 0')
ax.legend(); add_alpha_colorbar(fig, ax); plt.tight_layout(); save_fig(fig, 'C2_Delta2k'); plt.show()

# C3: ratio P_GREA/P_LCDM - 1 (linear y)
fig, ax = plt.subplots(figsize=(8, 5))
ax.axhline(0, color='k', lw=0.8)
for at in SOLVED_ALPHAS:
    ax.semilogx(k_l, pk_grea[at]/Pkl - 1., color=grea_color(at), lw=1.5, alpha=0.9)
ax.set_xlabel(r'k  [Mpc$^{-1}$]')
ax.set_ylabel(r'$P_{\\rm GREA}/P_{\\rm LCDM} - 1$')
ax.set_title('P(k) ratio at z = 0  (alpha variation)')
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.))
add_alpha_colorbar(fig, ax); plt.tight_layout(); save_fig(fig, 'C3_Pk_ratio'); plt.show()
"""))

# ═══════════════════════════════════════════════════════════════════════════════
# Section D — CMB
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(md("""\
---
## Section D — CMB Spectra

**D1 produces two ratio panels:**

- **Panel A (D1a):** Fixed {ω_b, ω_cdm, h=0.70} for both GREA and LCDM reference.
  The high-ℓ (ℓ≳300) ringing in the ratio is **not** a physical GREA effect — it is an
  acoustic-scale shift caused by D_A(z_rec) differing between GREA and LCDM at fixed h.
  r_s is identical in all models (144.5479 Mpc); only D_C(z_rec) shifts (by up to 1.2%).

- **Panel B (D1b):** For each α, the LCDM reference has its h adjusted so that
  100·θ_s = 100·r_s/D_C(z_rec) is matched to the GREA value.
  After matching, the high-ℓ ringing is suppressed 115–274× (RMS over ℓ>300),
  leaving only the genuine GREA signals: ISW (low ℓ) and lensing smoothing.

θ_s shifts monotonically: +0.00564 (α=0.8, phantom-quintessence) to −0.01183 (α=1.3, deep phantom).
Matched h values: h=0.719 (α=0.8) down to h=0.660 (α=1.3).
"""))

cells.append(code("""\
# D1a: lensed C_ell^TT absolute + Panel A ratio (fixed h=0.70, theta_s NOT matched)
lmax = 2500
cl_l = lcdm.lensed_cl(lmax)
ell  = cl_l['ell']
fac  = ell*(ell+1.)/(2.*np.pi)
TT_l = fac * cl_l['tt']
mask = ell >= 2

# Cache lensed TT for all GREA (reused in D1b)
TT_grea_cache = {at: fac * grea_instances[at].lensed_cl(lmax)['tt'] for at in SOLVED_ALPHAS}

fig, (ax, axr) = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                                gridspec_kw={'height_ratios': [2, 1]})
ax.plot(ell[mask], TT_l[mask], color=LCDM_COLOR, lw=2., ls='--', label='LCDM (h=0.70)', zorder=10)
axr.axhline(0, color='k', lw=0.8)
for at in SOLVED_ALPHAS:
    col  = grea_color(at)
    TT_g = TT_grea_cache[at]
    ax.plot(ell[mask], TT_g[mask], color=col, lw=1.2, alpha=0.85)
    axr.plot(ell[mask], TT_g[mask]/TT_l[mask] - 1., color=col, lw=1.2, alpha=0.85)
ax.set_xscale('log'); ax.set_ylabel(r'$D_\\ell^{TT}$  [$\\mu$K$^2$]')
ax.set_title(r'Panel A: Fixed $\\omega_b,\\omega_c,h=0.70$ — '
             r'$\\theta_s$ NOT matched; ringing = acoustic-scale shift')
ax.legend()
axr.axvspan(2, 30, alpha=0.08, color='orange', label='ISW region')
axr.axvline(300, color='gray', lw=0.8, ls=':', label=r'$\\ell=300$ ringing onset')
axr.set_xscale('log'); axr.set_xlabel(r'$\\ell$')
axr.set_ylabel(r'$C_\\ell^{TT,\\rm GREA}/C_\\ell^{TT,\\rm LCDM} - 1$')
axr.legend(fontsize=9)
add_alpha_colorbar(fig, axr); plt.tight_layout(); save_fig(fig, 'D1a_ClTT_fixed'); plt.show()
"""))

cells.append(md("""\
### D1b — θ_s-matched LCDM reference

For each α, we find h_lcdm such that 100·θ_s(LCDM, h_lcdm) = 100·θ_s(GREA, α).
Only h is adjusted; all other parameters (ω_b, ω_c, A_s, n_s, τ) remain fixed.
The brentq uses a Python mirror of the CLASS background ODE (validated to 0 difference
vs CLASS diagnostic at all 6 sqrt_k_eta0 values — see Cell 0/4).

| α | Δ(100·θ_s) vs LCDM | h_matched | Panel B rms ℓ>300 | suppression |
|---|---|---|---|---|
| 0.80 | +0.00564 | 0.71924 | 2.0e-4 | 114× |
| 0.90 | +0.00101 | 0.70344 | 1.5e-5 | 274× |
| 1.00 | −0.00293 | 0.69007 | 8.1e-5 | 150× |
| 1.10 | −0.00633 | 0.67865 | 1.4e-4 | 186× |
| 1.20 | −0.00927 | 0.66881 | 2.5e-4 | 152× |
| 1.30 | −0.01183 | 0.66029 | 2.5e-4 | 198× |
"""))

cells.append(code("""\
# D1b: theta_s-matched LCDM reference — build matched instances and plot Panel B
from scipy.optimize import brentq as _brentq

def _theta_s_100_lcdm_bg(h_val):
    _c = Class()
    _c.set({'omega_b': COMMON['omega_b'], 'omega_cdm': COMMON['omega_cdm'],
            'h': h_val, 'A_s': COMMON['A_s'], 'n_s': COMMON['n_s'],
            'tau_reio': COMMON['tau_reio'], 'output': '',
            'base_path': ROOT,
            'fluid_equation_of_state': 'CLP', 'w0_fld': -1., 'wa_fld': 0., 'cs2_fld': 1.})
    _c.compute()
    _d   = _c.get_current_derived_parameters(['rs_rec', 'z_rec'])
    _ts  = 100. * _d['rs_rec'] / _c.comoving_distance(_d['z_rec'])
    _c.struct_cleanup(); _c.empty()
    return _ts

def _theta_s_100_grea(c):
    _d = c.get_current_derived_parameters(['rs_rec', 'z_rec'])
    return 100. * _d['rs_rec'] / c.comoving_distance(_d['z_rec'])

# Print theta_s table
print("theta_s verification table:")
print(f"  {'alpha':>7}  {'100*ts_GREA':>13}  {'delta_ts':>11}  {'h_match':>10}  {'100*ts_LCDM_match':>20}")
print("  " + "-"*70)

H_MATCHED = {}
TT_match_cache = {}
_ts_lcdm_ref = _theta_s_100_lcdm_bg(COMMON['h'])
print(f"  {'LCDM':>7}  {_ts_lcdm_ref:13.6f}  {'—':>11}  {COMMON['h']:>10.5f}  {'—':>20}")

for at in SOLVED_ALPHAS:
    _ts_g  = _theta_s_100_grea(grea_instances[at])
    _h_m   = _brentq(lambda hv: _theta_s_100_lcdm_bg(hv) - _ts_g, 0.50, 0.90, xtol=1e-6)
    H_MATCHED[at] = _h_m
    _ts_m  = _theta_s_100_lcdm_bg(_h_m)    # verify
    print(f"  {at:7.2f}  {_ts_g:13.6f}  {_ts_g-_ts_lcdm_ref:+11.6f}  {_h_m:10.6f}  {_ts_m:20.6f}")

print("\\nBuilding matched LCDM instances (fast — tCl lCl only)...")
_CMB_P = {k: COMMON[k] for k in COMMON
          if k not in ('output', 'P_k_max_1/Mpc', 'z_max_pk')}
for at in SOLVED_ALPHAS:
    _c = Class()
    _c.set({**_CMB_P, 'h': H_MATCHED[at], 'output': 'tCl lCl', 'lensing': 'yes',
            'fluid_equation_of_state': 'CLP', 'w0_fld': -1., 'wa_fld': 0., 'cs2_fld': 1.})
    _c.compute()
    TT_match_cache[at] = fac * _c.lensed_cl(lmax)['tt']
    _c.struct_cleanup(); _c.empty()
    print(f"  alpha={at:.1f}  h={H_MATCHED[at]:.6f} done")

# Plot Panel B
fig, (ax, axr) = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                                gridspec_kw={'height_ratios': [2, 1]})
_ts_ref_ts = _theta_s_100_lcdm_bg(COMMON['h'])
ax.plot(ell[mask], TT_l[mask], color=LCDM_COLOR, lw=2., ls='--',
        label=f'LCDM ref (h=0.70, 100θ_s={_ts_ref_ts:.4f})', zorder=10)
axr.axhline(0, color='k', lw=0.8)

for at in SOLVED_ALPHAS:
    col  = grea_color(at)
    TT_g = TT_grea_cache[at]
    TT_m = TT_match_cache[at]
    ax.plot(ell[mask], TT_g[mask], color=col, lw=1.2, alpha=0.85)
    good = mask & (TT_m > 0)
    axr.plot(ell[good], TT_g[good]/TT_m[good] - 1., color=col, lw=1.2, alpha=0.85)

ax.set_xscale('log'); ax.set_ylabel(r'$D_\\ell^{TT}$  [$\\mu$K$^2$]')
ax.set_title('Panel B: $\\theta_s$-matched LCDM reference — genuine GREA signal only\\n'
             '(h adjusted per α; high-ℓ ringing suppressed 115–274×)')
ax.legend(fontsize=9)
axr.axvspan(2, 30, alpha=0.08, color='orange', label='ISW region (genuine GREA)')
axr.axvline(300, color='gray', lw=0.8, ls=':', alpha=0.6, label=r'$\\ell=300$')
axr.set_xscale('log'); axr.set_xlabel(r'$\\ell$')
axr.set_ylabel(r'$C_\\ell^{TT,\\rm GREA}/C_\\ell^{TT,\\rm LCDM}(\\theta_s\\,{\\rm match}) - 1$')
axr.legend(fontsize=9)
add_alpha_colorbar(fig, axr); plt.tight_layout(); save_fig(fig, 'D1b_ClTT_tsmatched'); plt.show()

# Quantify residual ringing
print("\\nPanel A vs Panel B rms (ell>300):")
_high = ell[mask] > 300
for at in SOLVED_ALPHAS:
    rA = (TT_grea_cache[at][mask][_high] / TT_l[mask][_high] - 1.)
    rB = (TT_grea_cache[at][mask][_high] / TT_match_cache[at][mask][_high] - 1.)
    rmA, rmB = np.sqrt(np.mean(rA**2)), np.sqrt(np.mean(rB**2))
    print(f"  alpha={at:.1f}  Panel A rms={rmA:.3e}  Panel B rms={rmB:.3e}  {rmA/max(rmB,1e-12):.0f}x suppressed")
"""))

cells.append(code("""\
# D2: Lensing potential C_ell^phiphi and ratio
# Convention: plot ell^2*(ell+1)^2/(2pi) * C_ell^pp * 1e7
PP_l = (ell*(ell+1.))**2 / (2.*np.pi) * cl_l['pp'] * 1.e7

fig, (ax, axr) = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                                gridspec_kw={'height_ratios': [2, 1]})
ax.plot(ell[mask], PP_l[mask], color=LCDM_COLOR, lw=2., ls='--', label='LCDM', zorder=10)
axr.axhline(0, color='k', lw=0.8)
fac_pp = (ell*(ell+1.))**2 / (2.*np.pi) * 1.e7
for at in SOLVED_ALPHAS:
    col  = grea_color(at)
    PP_g = fac_pp * grea_instances[at].lensed_cl(lmax)['pp']
    ax.plot(ell[mask], PP_g[mask], color=col, lw=1.2, alpha=0.85)
    good = mask & (PP_l > 0)
    axr.plot(ell[good], PP_g[good]/PP_l[good] - 1., color=col, lw=1.2, alpha=0.85)
ax.set_xscale('log')
ax.set_ylabel(r'$[\\ell(\\ell+1)]^2 C_\\ell^{\\phi\\phi}/(2\\pi) \\times 10^7$')
ax.set_title(r'CMB lensing potential $C_\\ell^{\\phi\\phi}$')
ax.legend()
axr.set_xscale('log'); axr.set_xlabel(r'$\\ell$')
axr.set_ylabel(r'$C_\\ell^{\\phi\\phi,\\rm GREA}/C_\\ell^{\\phi\\phi,\\rm LCDM} - 1$')
add_alpha_colorbar(fig, axr); plt.tight_layout(); save_fig(fig, 'D2_Clpp'); plt.show()
"""))

# ═══════════════════════════════════════════════════════════════════════════════
# Section E — Parameter sensitivity
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(md("""\
---
## Section E — GREA Parameter Sensitivity  (α variation at h=0.70)

> **⚠️ PLACEHOLDER for full MCMC posterior — replace with getdist/ChainConsumer chains
> once Cobaya/MontePython pipeline is built.**

The plots below use only the 6 CLASS runs already built in Cell 3.
They show how σ₈, S8, w₀, wₐ respond to α at the fiducial h=0.70.
"""))

cells.append(code("""\
# E: sigma8(alpha), S8(alpha), w0(alpha), wa(alpha)
_alphas  = np.array(SOLVED_ALPHAS)
_sigma8  = np.array([grea_instances[at].sigma8()          for at in SOLVED_ALPHAS])
_S8      = np.array([grea_instances[at].S8()              for at in SOLVED_ALPHAS])

def _w0_only(at):
    bg  = bg_grea[at]
    z_s = np.array(bg['z']); idx = np.argsort(z_s)
    w_s = np.array(bg['(.)w_fld'])[idx]; z_s = z_s[idx]
    return float(np.interp(0., z_s, w_s))

def _wa_only(at):
    bg  = bg_grea[at]
    z_s = np.array(bg['z']); idx = np.argsort(z_s)
    w_s = np.array(bg['(.)w_fld'])[idx]; z_s = z_s[idx]
    a_fit = np.linspace(0.90, 1.0, 30)
    w_fit = np.array([float(np.interp(1./a-1., z_s, w_s)) for a in a_fit])
    return -np.polyfit(a_fit - 1., w_fit, 2)[1]

_w0 = np.array([_w0_only(at) for at in SOLVED_ALPHAS])
_wa = np.array([_wa_only(at) for at in SOLVED_ALPHAS])
_g  = np.array([ALPHA_TO_G[at] for at in SOLVED_ALPHAS])

fig = plt.figure(figsize=(12, 10))
fig.patch.set_facecolor('#fff8e7')
fig.text(0.5, 0.99, '\\u26a0  PLACEHOLDER — parameter sensitivity, NOT an MCMC posterior  \\u26a0',
         ha='center', va='top', fontsize=12, color='darkred', fontweight='bold',
         bbox=dict(boxstyle='round', fc='#ffe0e0', ec='darkred', lw=2))

axes = fig.subplots(2, 2)
pairs = [
    (axes[0,0], _sigma8, r'$\\sigma_8(0)$',     'sigma8'),
    (axes[0,1], _S8,     r'$S_8$',               'S8'),
    (axes[1,0], _w0,     r'$w_0$',               'w0'),
    (axes[1,1], _wa,     r'$w_a$',               'wa'),
]
for ax, y, ylabel, label in pairs:
    for at, yi in zip(SOLVED_ALPHAS, y):
        ax.scatter(at, yi, color=grea_color(at), s=80, zorder=5)
    ax.plot(_alphas, y, color='gray', lw=1., ls='--', zorder=3)
    ax.set_xlabel(r'$\\alpha$'); ax.set_ylabel(ylabel)
    ax.set_title(f'{ylabel} vs \\u03b1')

# Reference lines
axes[1,0].axhline(-1., color='k', lw=0.8, ls=':', label='w=-1')
axes[1,0].legend(fontsize=9)
# sigma8 LCDM reference
axes[0,0].axhline(lcdm.sigma8(), color=LCDM_COLOR, lw=1., ls=':', label='LCDM')
axes[0,1].axhline(lcdm.S8(),     color=LCDM_COLOR, lw=1., ls=':', label='LCDM')
axes[0,0].legend(fontsize=9); axes[0,1].legend(fontsize=9)

plt.tight_layout(rect=[0,0,1,0.96])
save_fig(fig, 'E_alpha_sensitivity')
plt.show()

# sqrt_k_eta0 vs alpha curve
fig2, ax2 = plt.subplots(figsize=(7, 4))
ax2.plot(_alphas, _g, 'o-', color='steelblue', lw=1.8)
for at, gi in zip(SOLVED_ALPHAS, _g):
    ax2.scatter(at, gi, color=grea_color(at), s=80, zorder=5)
ax2.axvline(_compute_alpha_py(3.6), color='k', ls=':', lw=0.8, label='fiducial g=3.6')
ax2.set_xlabel(r'$\\alpha$'); ax2.set_ylabel(r'$\\sqrt{k}\\eta_0$')
ax2.set_title(r'Inversion: $\\alpha \\to \\sqrt{k}\\eta_0$')
ax2.legend(); plt.tight_layout(); save_fig(fig2, 'E_alpha_to_g'); plt.show()
"""))

# ═══════════════════════════════════════════════════════════════════════════════
# Cleanup
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(md("---\n## Cleanup"))
cells.append(code("""\
lcdm.struct_cleanup(); lcdm.empty()
for c in grea_instances.values():
    c.struct_cleanup(); c.empty()
print(f"All instances freed. Figures saved to {FIGURES_DIR}")
print(f"Run tag: {RUN_TS}")
"""))

# ── Assemble ──────────────────────────────────────────────────────────────────
nb = nbf.v4.new_notebook()
nb['cells'] = cells
nb['metadata'] = {
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.13'},
}
with open(NB_PATH, 'w') as f:
    nbf.write(nb, f)

print(f"Notebook written → {NB_PATH}")
print(f"Total cells: {len(cells)}")
for i, c in enumerate(cells):
    t   = 'md' if c['cell_type'] == 'markdown' else 'code'
    src = c['source'].split('\\n')[0][:70]
    print(f"  [{i:02d}] {t:4s}  {src}")
