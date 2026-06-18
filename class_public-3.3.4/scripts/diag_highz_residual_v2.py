#!/usr/bin/env python3
"""
Read-only diagnostic v2: correct high-z residual check.

The Friedmann equation gives:
  H_GREA²(z) - H_LCDM²(z) = rho_fld_GREA(z) - rho_fld_LCDM(z)

  =>  H_GREA/H_LCDM - 1  ≈  ½·(Omega_fld_GREA - Omega_fld_LCDM)
                           (to leading order in the small difference)

At z >> 100, Omega_fld_LCDM → 0 and the formula reduces to ½·Omega_fld_GREA.
At z = 10–50, LCDM still has non-negligible Omega_Lambda ≈ 1–2e-3, so the
naive ½·Omega_fld_GREA test overpredicts the expected H difference.

This script computes both the naive ratio AND the corrected ratio.
No source edits. No file modifications.
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
    'base_path': ROOT, 'output': '',
}
LCDM_P = {**PARAMS, 'fluid_equation_of_state': 'CLP',
           'w0_fld': -1.0, 'wa_fld': 0.0, 'cs2_fld': 1.0}
GREA_P = {**PARAMS, 'fluid_equation_of_state': 'GREA',
           'sqrt_k_eta0': 3.6, 'cs2_fld': 1.0, 'use_ppf': 'yes'}

def run(p):
    c = Class(); c.set(p); c.compute(); return c

lcdm = run(LCDM_P)
grea = run(GREA_P)

bg_l = lcdm.get_background()
bg_g = grea.get_background()

def sorted_arr(bg, key):
    z = np.array(bg['z'])
    idx = np.argsort(z)
    return np.array(bg['z'])[idx], np.array(bg[key])[idx]

z_l, H_l        = sorted_arr(bg_l, 'H [1/Mpc]')
z_l, rc_l       = sorted_arr(bg_l, '(.)rho_crit')
z_g, H_g        = sorted_arr(bg_g, 'H [1/Mpc]')
z_g, rc_g       = sorted_arr(bg_g, '(.)rho_crit')
z_g, rho_fld_g  = sorted_arr(bg_g, '(.)rho_fld')

# LCDM fluid density: CLP w0=-1 stores fluid in rho_fld (not rho_lambda)
lcdm_keys = list(bg_l.keys())
if '(.)rho_fld' in lcdm_keys:
    z_l, rho_fld_l = sorted_arr(bg_l, '(.)rho_fld')
    lcdm_fluid_key = '(.)rho_fld'
elif '(.)rho_lambda' in lcdm_keys:
    z_l, rho_fld_l = sorted_arr(bg_l, '(.)rho_lambda')
    lcdm_fluid_key = '(.)rho_lambda'
else:
    # Reconstruct from rho_crit - rho_m - rho_r
    z_l2, rho_b  = sorted_arr(bg_l, '(.)rho_b')
    z_l2, rho_c  = sorted_arr(bg_l, '(.)rho_cdm')
    z_l2, rho_g  = sorted_arr(bg_l, '(.)rho_g')
    z_l2, rho_ur = sorted_arr(bg_l, '(.)rho_ur') if '(.)rho_ur' in lcdm_keys else (z_l2, np.zeros_like(z_l2))
    rho_fld_l = rc_l - rho_b - rho_c - rho_g - rho_ur
    lcdm_fluid_key = 'rho_crit - rho_m - rho_r (computed)'

print(f"LCDM fluid density key: {lcdm_fluid_key}")
print(f"GREA fluid density key: (.)rho_fld")
print()

Z_PROBE = [10.0, 50.0, 200.0, 1000.0]

print("=" * 110)
print("CORRECTED HIGH-z RESIDUAL CHECK")
print("  Expected: (H_GREA/H_LCDM - 1) / [½·(Omega_fld_GREA - Omega_fld_LCDM)] ≈ 1.0 ± 0.10")
print("=" * 110)
print(f"{'z':>7}  {'dH/H':>12}  {'Omf_G':>12}  {'Omf_L':>12}  "
      f"{'dOmf':>12}  {'½dOmf':>12}  "
      f"{'naive ratio':>12}  {'corr ratio':>12}  verdict")
print("-" * 110)

for z in Z_PROBE:
    Hl  = float(np.interp(z, z_l, H_l))
    Hg  = float(np.interp(z, z_g, H_g))
    dHH = Hg / Hl - 1.0

    rcl = float(np.interp(z, z_l, rc_l))
    rcg = float(np.interp(z, z_g, rc_g))
    rfl = float(np.interp(z, z_l, rho_fld_l))
    rfg = float(np.interp(z, z_g, rho_fld_g))

    Omf_G = rfg / rcg
    Omf_L = rfl / rcl
    dOmf  = Omf_G - Omf_L

    naive_ratio = dHH / (0.5 * Omf_G) if abs(Omf_G) > 1e-30 else float('nan')
    corr_ratio  = dHH / (0.5 * dOmf)  if abs(dOmf)  > 1e-30 else float('nan')

    if abs(corr_ratio - 1.0) < 0.10:
        verdict = "PHYSICAL ✓"
    elif abs(corr_ratio - 1.0) < 0.30:
        verdict = "close"
    else:
        verdict = "ANOMALOUS ←"

    print(f"{z:7.1f}  {dHH:+12.4e}  {Omf_G:12.4e}  {Omf_L:12.4e}  "
          f"{dOmf:+12.4e}  {0.5*dOmf:+12.4e}  "
          f"{naive_ratio:12.4f}  {corr_ratio:12.4f}  {verdict}")

print()
print("Columns:")
print("  dH/H     = H_GREA/H_LCDM - 1")
print("  Omf_G    = Omega_fld_GREA(z) = rho_fld_GREA / rho_crit_GREA")
print("  Omf_L    = Omega_fld_LCDM(z) = rho_fld_LCDM / rho_crit_LCDM  (w=-1 fluid)")
print("  dOmf     = Omf_G - Omf_L")
print("  naive    = dH/H / (½·Omf_G)   [ignores LCDM fluid]")
print("  corr     = dH/H / (½·dOmf)    [correct comparison]")

lcdm.struct_cleanup(); lcdm.empty()
grea.struct_cleanup(); grea.empty()
print("\nDone.")
