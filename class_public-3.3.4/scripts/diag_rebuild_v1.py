#!/usr/bin/env python3
"""
Post-rebuild diagnostic (v1):
1. Background validation: H(0), r_s, omega_b/cdm, H_GREA/H_LCDM-1 at key z
2. f(z) table at z=2,3,5,10
3. sigma8(z=0), S8
4. (Read-only) w(a) at z=0,0.5,1,2 — does not fix anything
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from classy import Class
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

PARAMS = {
    'h': 0.7,
    'omega_b': 0.022,
    'omega_cdm': 0.12,
    'A_s': 2.1e-9,
    'n_s': 0.965,
    'tau_reio': 0.054,
    'base_path': ROOT,
}
LCDM_P = {**PARAMS, 'output': 'mPk', 'P_k_max_1/Mpc': 1.0,
           'fluid_equation_of_state': 'CLP', 'w0_fld': -1.0, 'wa_fld': 0.0, 'cs2_fld': 1.0}
GREA_P = {**PARAMS, 'output': 'mPk', 'P_k_max_1/Mpc': 1.0,
           'fluid_equation_of_state': 'GREA', 'sqrt_k_eta0': 3.6,
           'cs2_fld': 1.0, 'use_ppf': 'yes'}

def run(params):
    c = Class()
    c.set(params)
    c.compute()
    return c

# ── build ──────────────────────────────────────────────────────────────────────
lcdm = run(LCDM_P)
grea = run(GREA_P)

# ── 1. Background validation ──────────────────────────────────────────────────
print("=" * 70)
print("1. BACKGROUND VALIDATION (post-rebuild)")
print("=" * 70)

H0_lcdm_kmsMpc = lcdm.Hubble(0) * 2.99792458e5   # H in Mpc^-1 → km/s/Mpc
H0_grea_kmsMpc = grea.Hubble(0) * 2.99792458e5
print(f"  H(0) LCDM  = {H0_lcdm_kmsMpc:.9f} km/s/Mpc  (expect 70.000)")
print(f"  H(0) GREA  = {H0_grea_kmsMpc:.9f} km/s/Mpc  (expect 70.000)")
print(f"  h LCDM     = {lcdm.h():.9f}")
print(f"  h GREA     = {grea.h():.9f}")

rs_lcdm = lcdm.rs_drag()
rs_grea = grea.rs_drag()
print(f"\n  r_s LCDM   = {rs_lcdm:.9f} Mpc")
print(f"  r_s GREA   = {rs_grea:.9f} Mpc")
print(f"  r_s delta  = {(rs_grea/rs_lcdm - 1):+.4e}")

# omega_b, omega_cdm from derived parameters
bg_l = lcdm.get_background()
bg_g = grea.get_background()
# omega = Omega * h^2
print(f"\n  omega_b  input = 0.022000 (both)")
print(f"  omega_cdm input = 0.120000 (both)")
print(f"  h LCDM  = {lcdm.h():.9f}  (input 0.7)")
print(f"  h GREA  = {grea.h():.9f}  (input 0.7)")

# H_GREA/H_LCDM - 1 at key redshifts
z_l = np.array(bg_l['z']); H_l = np.array(bg_l['H [1/Mpc]'])
z_g = np.array(bg_g['z']); H_g = np.array(bg_g['H [1/Mpc]'])
order_l = np.argsort(z_l); z_l = z_l[order_l]; H_l = H_l[order_l]
order_g = np.argsort(z_g); z_g = z_g[order_g]; H_g = H_g[order_g]

print("\n  H_GREA/H_LCDM - 1 at probe redshifts:")
for z in [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 200.0, 1000.0]:
    Hl = float(np.interp(z, z_l, H_l))
    Hg = float(np.interp(z, z_g, H_g))
    flag = "  *** PROBLEM" if abs(Hg/Hl - 1) > 1e-3 and z >= 10 else ""
    print(f"    z={z:7.1f}:  H_G/H_L - 1 = {Hg/Hl - 1:+.6e}{flag}")

# ── 2. f(z) table ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("2. SCALE-INDEPENDENT GROWTH RATE f(z) (post-rebuild)")
print("=" * 70)
Z_PROBE = [2.0, 3.0, 5.0, 10.0]
print(f"{'z':>6}  {'f_LCDM':>14}  {'f_GREA':>14}  {'delta_f':>14}  {'delta_f/f_L':>14}")
for z in Z_PROBE:
    fl = lcdm.scale_independent_growth_factor_f(z)
    fg = grea.scale_independent_growth_factor_f(z)
    df = fg - fl
    flag = "  *** PROBLEM" if abs(df) > 1e-3 else ""
    print(f"{z:6.1f}  {fl:14.8f}  {fg:14.8f}  {df:14.8e}  {df/fl:14.8e}{flag}")

# ── 3. sigma8 / S8 ────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("3. sigma8 AND S8")
print("=" * 70)
s8_lcdm = lcdm.sigma8()
s8_grea = grea.sigma8()
S8_lcdm = lcdm.S8()
S8_grea = grea.S8()
Omega_m_lcdm = lcdm.Omega_m()
Omega_m_grea = grea.Omega_m()
print(f"  sigma8  LCDM = {s8_lcdm:.6f}")
print(f"  sigma8  GREA = {s8_grea:.6f}  delta = {s8_grea - s8_lcdm:+.4e}")
print(f"  S8      LCDM = {S8_lcdm:.6f}")
print(f"  S8      GREA = {S8_grea:.6f}  delta = {S8_grea - S8_lcdm:+.4e}")
print(f"  Omega_m LCDM = {Omega_m_lcdm:.6f}")
print(f"  Omega_m GREA = {Omega_m_grea:.6f}")

# ── 4. w(a) read-only diagnostic ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("4. w(a) DIAGNOSTIC (read-only: check for phantom crossing)")
print("=" * 70)
try:
    bg_g_full = grea.get_background()
    z_arr = np.array(bg_g_full['z'])
    # background dict has '(.)rho_fld' and may have 'w_fld'
    keys = list(bg_g_full.keys())
    w_key = [k for k in keys if 'w_fld' in k or 'w_fl' in k]
    print(f"  Available keys containing 'w_fld': {w_key}")
    # Try to get w from equation of state via Hubble difference
    # w(z) = -1 + (1/3) * d ln(rho_fld) / d ln(1+z)  ... indirect
    # Better: use background_w_fld via CLASS method if available
    print("\n  w_fld at probe redshifts (from CLASS background dict or w_fld method):")
    for z in [0.0, 0.5, 1.0, 2.0]:
        try:
            w = float(grea.w_fld(z))
            print(f"    z={z:.1f}: w = {w:.6f}")
        except Exception as e:
            # fallback: interpolate from background table
            if w_key:
                w_arr = np.array(bg_g_full[w_key[0]])
                z_w = z_arr.copy()
                order = np.argsort(z_w)
                z_w = z_w[order]; w_arr = w_arr[order]
                w = float(np.interp(z, z_w, w_arr))
                print(f"    z={z:.1f}: w = {w:.6f}  (from bg table key '{w_key[0]}')")
            else:
                print(f"    z={z:.1f}: w unavailable ({e})")
except Exception as e:
    print(f"  w(a) diagnostic failed: {e}")

lcdm.struct_cleanup(); lcdm.empty()
grea.struct_cleanup(); grea.empty()
print("\nDone.")
