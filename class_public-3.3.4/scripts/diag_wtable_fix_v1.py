#!/usr/bin/env python3
"""
Acceptance test for the w_table physical-E fix (gated step).

1. w(a) trajectory at z=0,0.25,0.5,1,1.5,2,3: print w0, wa, phantom crossing z
2. Background regression: H(0), r_s vs LCDM
3. sigma8, S8 regression (must be unchanged within 1e-4 of previous build values)
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from classy import Class
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

PARAMS = {
    'h': 0.7, 'omega_b': 0.022, 'omega_cdm': 0.12,
    'A_s': 2.1e-9, 'n_s': 0.965, 'tau_reio': 0.054,
    'base_path': ROOT,
}
LCDM_P = {**PARAMS, 'output': 'mPk', 'P_k_max_1/Mpc': 1.0,
           'fluid_equation_of_state': 'CLP', 'w0_fld': -1.0, 'wa_fld': 0.0, 'cs2_fld': 1.0}
GREA_P = {**PARAMS, 'output': 'mPk', 'P_k_max_1/Mpc': 1.0,
           'fluid_equation_of_state': 'GREA', 'sqrt_k_eta0': 3.6,
           'cs2_fld': 1.0, 'use_ppf': 'yes'}

def run(p):
    c = Class(); c.set(p); c.compute(); return c

lcdm = run(LCDM_P)
grea = run(GREA_P)

bg_g = grea.get_background()
bg_l = lcdm.get_background()

def sorted_arr(bg, *keys):
    z = np.array(bg['z']); idx = np.argsort(z); z = z[idx]
    return (z,) + tuple(np.array(bg[k])[idx] for k in keys)

z_g, w_g = sorted_arr(bg_g, '(.)w_fld')
z_l, H_l = sorted_arr(bg_l, 'H [1/Mpc]')
z_g2, H_g = sorted_arr(bg_g, 'H [1/Mpc]')

# ── 1. w(a) trajectory ────────────────────────────────────────────────────────
print("=" * 70)
print("1. w(a) TRAJECTORY (acceptance test: physical E fix)")
print("=" * 70)
Z_W = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
w_vals = []
print(f"  {'z':>6}  {'a':>8}  {'w(z)':>12}")
for z in Z_W:
    w = float(np.interp(z, z_g, w_g))
    a = 1.0 / (1.0 + z)
    w_vals.append(w)
    print(f"  {z:6.2f}  {a:8.4f}  {w:12.6f}")

# w0 = w(z=0)
w0 = w_vals[0]

# wa = -dw/da |_{a=1}  estimated from finite difference around a=1
# Use z=0 and z=0.25 (a=1.0 and a=0.8)
a0  = 1.0;  w_a0 = w_vals[0]
a1  = 1.0 / 1.25;  w_a1 = float(np.interp(0.25, z_g, w_g))
wa_num = -(w_a0 - w_a1) / (a0 - a1)   # -dw/da at a~1

# Also estimate with z=0 and z=0.5 (a=1.0 and a=0.667)
a2 = 1.0 / 1.5; w_a2 = float(np.interp(0.5, z_g, w_g))
wa_num2 = -(w_a0 - w_a2) / (a0 - a2)

print()
print(f"  w0 = w(z=0)       = {w0:.6f}   (expect ≈ -1.0)")
print(f"  wa ≈ -dw/da|a=1   = {wa_num:.4f}   [from z=0,0.25]")
print(f"  wa ≈ -dw/da|a=1   = {wa_num2:.4f}   [from z=0,0.5]")
print(f"  (García-Bellido 2024 expect: w0 ≈ -1, wa ≈ -0.3)")

# Phantom crossing: find z where w crosses -1 (from above, i.e. w goes below -1)
# The bg table: find sign changes in (w+1)
w_plus1 = w_g + 1.0
sign_changes = np.where(np.diff(np.sign(w_plus1)))[0]
if len(sign_changes) > 0:
    # Linear interpolate to find exact crossing
    crossings = []
    for sc in sign_changes:
        z_lo, z_hi = z_g[sc], z_g[sc + 1]
        w_lo, w_hi = w_plus1[sc], w_plus1[sc + 1]
        z_cross = z_lo + (0 - w_lo) * (z_hi - z_lo) / (w_hi - w_lo)
        crossings.append(z_cross)
    print(f"\n  Phantom crossing(s) at z = {[f'{zc:.4f}' for zc in crossings]}")
    print(f"  (expect z ≲ 2 for transient crossing)")
else:
    print(f"\n  No phantom crossing detected in background table")
    # Report range
    print(f"  w range: [{w_g.min():.6f}, {w_g.max():.6f}]")

# ── 2. Background regression ──────────────────────────────────────────────────
print()
print("=" * 70)
print("2. BACKGROUND REGRESSION")
print("=" * 70)
H0_lcdm_kms = lcdm.Hubble(0) * 2.99792458e5
H0_grea_kms = grea.Hubble(0) * 2.99792458e5
rs_lcdm = lcdm.rs_drag()
rs_grea = grea.rs_drag()

print(f"  H(0) LCDM  = {H0_lcdm_kms:.9f} km/s/Mpc")
print(f"  H(0) GREA  = {H0_grea_kms:.9f} km/s/Mpc  delta = {(H0_grea_kms/H0_lcdm_kms - 1):+.4e}")
print(f"  r_s LCDM   = {rs_lcdm:.9f} Mpc")
print(f"  r_s GREA   = {rs_grea:.9f} Mpc  delta = {(rs_grea/rs_lcdm - 1):+.4e}")

# H_GREA/H_LCDM at high z (regression check)
print("\n  H_GREA/H_LCDM - 1 at key z:")
for z in [0.0, 1.0, 2.0, 5.0, 10.0, 50.0, 1000.0]:
    Hl = float(np.interp(z, z_l, H_l))
    Hg = float(np.interp(z, z_g2, H_g))
    print(f"    z={z:7.1f}:  {Hg/Hl - 1:+.6e}")

# ── 3. sigma8 / S8 regression ────────────────────────────────────────────────
print()
print("=" * 70)
print("3. sigma8 / S8 REGRESSION")
print("=" * 70)
# Previous (pre-fix) values from diag_rebuild_v1.py:
#   sigma8 LCDM = 0.832417, GREA = 0.847281
#   S8     LCDM = 0.818138, GREA = 0.832747
prev_s8_lcdm = 0.832417; prev_s8_grea = 0.847281
prev_S8_lcdm = 0.818138; prev_S8_grea = 0.832747

s8_lcdm = lcdm.sigma8(); s8_grea = grea.sigma8()
S8_lcdm = lcdm.S8();     S8_grea = grea.S8()

print(f"  {'':12}  {'LCDM':>12}  {'GREA':>12}  {'LCDM prev':>12}  {'GREA prev':>12}  {'LCDM chg':>12}  {'GREA chg':>12}")
print(f"  {'sigma8':12}  {s8_lcdm:12.6f}  {s8_grea:12.6f}  {prev_s8_lcdm:12.6f}  {prev_s8_grea:12.6f}  "
      f"{s8_lcdm-prev_s8_lcdm:+12.4e}  {s8_grea-prev_s8_grea:+12.4e}")
print(f"  {'S8':12}  {S8_lcdm:12.6f}  {S8_grea:12.6f}  {prev_S8_lcdm:12.6f}  {prev_S8_grea:12.6f}  "
      f"{S8_lcdm-prev_S8_lcdm:+12.4e}  {S8_grea-prev_S8_grea:+12.4e}")

thresh = 1e-4
ok_s8_l = abs(s8_lcdm - prev_s8_lcdm) < thresh
ok_s8_g = abs(s8_grea - prev_s8_grea) < thresh
ok_S8_l = abs(S8_lcdm - prev_S8_lcdm) < thresh
ok_S8_g = abs(S8_grea - prev_S8_grea) < thresh
all_ok  = ok_s8_l and ok_s8_g and ok_S8_l and ok_S8_g
print(f"\n  All sigma8/S8 changes < 1e-4: {'PASS ✓' if all_ok else 'FAIL ← REGRESSION'}")
if not all_ok:
    if not ok_s8_g:
        print(f"  WARNING: sigma8_GREA changed by {s8_grea-prev_s8_grea:+.4e} (> 1e-4)")
    if not ok_S8_g:
        print(f"  WARNING: S8_GREA changed by {S8_grea-prev_S8_grea:+.4e} (> 1e-4)")

# ── 4. f(z) regression ───────────────────────────────────────────────────────
print()
print("=" * 70)
print("4. f(z) REGRESSION (w_table should not touch growth — verify)")
print("=" * 70)
Z_F = [2.0, 3.0, 5.0, 10.0]
print(f"  {'z':>6}  {'f_LCDM':>14}  {'f_GREA':>14}  {'delta_f':>14}")
for z in Z_F:
    fl = lcdm.scale_independent_growth_factor_f(z)
    fg = grea.scale_independent_growth_factor_f(z)
    print(f"  {z:6.1f}  {fl:14.8f}  {fg:14.8f}  {fg-fl:+14.8e}")

lcdm.struct_cleanup(); lcdm.empty()
grea.struct_cleanup(); grea.empty()
print("\nDone.")
