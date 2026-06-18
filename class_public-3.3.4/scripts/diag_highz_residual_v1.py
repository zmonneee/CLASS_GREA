#!/usr/bin/env python3
"""
Read-only diagnostic: check whether H_GREA/H_LCDM-1 at high z
tracks ½·Omega_fld(z), as expected for a genuine physical GREA
dark-energy tail, or is anomalously large (second normalization bug).

Expected relationship: at high z where GREA DE is small,
  H² ≈ H_LCDM² + rho_fld  =>  H_GREA/H_LCDM - 1 ≈ ½ Omega_fld(z)
A ratio (H_GREA/H_LCDM-1) / (½ Omega_fld) = 1 ± 0.10 is physical.
A ratio >> 1 indicates a second residual normalization error.

No source edits. No file modifications.
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
    'output': '',       # background only
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

# sort by z ascending
def sorted_bg(bg, *keys):
    z = np.array(bg['z'])
    idx = np.argsort(z)
    z = z[idx]
    return (z,) + tuple(np.array(bg[k])[idx] for k in keys)

# LCDM: need H, rho_crit
z_l, H_l, rho_crit_l = sorted_bg(bg_l, 'H [1/Mpc]', '(.)rho_crit')

# GREA: need H, rho_crit, rho_fld, Omega_m
g_keys_avail = list(bg_g.keys())
print("Available GREA background keys:")
print([k for k in g_keys_avail if 'fld' in k or 'rho' in k or 'Omega' in k or 'crit' in k])
print()

# Determine the fluid density key
fld_rho_key = None
for candidate in ['(.)rho_fld', 'rho_fld', '(.)rho_de']:
    if candidate in g_keys_avail:
        fld_rho_key = candidate
        break

omega_fld_key = None
for candidate in ['Omega_fld(z)', 'Omega_de(z)', '(.)Omega_fld']:
    if candidate in g_keys_avail:
        omega_fld_key = candidate
        break

print(f"Using fluid density key : {fld_rho_key}")
print(f"Using Omega_fld key     : {omega_fld_key}")
print()

z_g, H_g, rho_crit_g = sorted_bg(bg_g, 'H [1/Mpc]', '(.)rho_crit')

# Omega_fld from background: either direct key or compute from rho
if omega_fld_key:
    _, Omega_fld_g = sorted_bg(bg_g, omega_fld_key)[0], sorted_bg(bg_g, omega_fld_key)[1]
    z_g2, Omega_fld_g = sorted_bg(bg_g, omega_fld_key)
    z_g2 = z_g  # already sorted above
    Omega_fld_g = np.array(bg_g[omega_fld_key])[np.argsort(np.array(bg_g['z']))]
elif fld_rho_key:
    rho_fld_g = np.array(bg_g[fld_rho_key])[np.argsort(np.array(bg_g['z']))]
    Omega_fld_g = rho_fld_g / rho_crit_g
    omega_fld_key = f"{fld_rho_key} / rho_crit  (computed)"
else:
    print("ERROR: no fluid density or Omega_fld key found in background output")
    lcdm.empty(); grea.empty()
    sys.exit(1)

# Also get Omega_m for context
_, Omega_m_g = sorted_bg(bg_g, 'Omega_m(z)')[0], None
Omega_m_g = np.array(bg_g['Omega_m(z)'])[np.argsort(np.array(bg_g['z']))]

Z_PROBE = [10.0, 50.0, 200.0, 1000.0]

print("=" * 90)
print("HIGH-z RESIDUAL DIAGNOSTIC")
print(f"  Test: is (H_GREA/H_LCDM - 1) ≈ ½·Omega_fld(z)?")
print(f"  Omega_fld key used: {omega_fld_key}")
print("=" * 90)
print(f"{'z':>8}  {'H_G/H_L-1':>14}  {'Omega_fld':>14}  {'½·Omega_fld':>14}  "
      f"{'ratio':>10}  {'Omega_m':>10}  {'verdict':>20}")
print("-" * 90)

for z in Z_PROBE:
    Hl  = float(np.interp(z, z_l, H_l))
    Hg  = float(np.interp(z, z_g, H_g))
    dHH = Hg / Hl - 1.0

    Omf = float(np.interp(z, z_g, Omega_fld_g))
    Omm = float(np.interp(z, z_g, Omega_m_g))
    half_Omf = 0.5 * Omf

    if abs(half_Omf) > 1e-30:
        ratio = dHH / half_Omf
    else:
        ratio = float('nan')

    if abs(ratio - 1.0) < 0.10:
        verdict = "PHYSICAL ✓"
    elif abs(ratio - 1.0) < 0.30:
        verdict = "close (~20%)"
    else:
        verdict = "ANOMALOUS ← investigate"

    print(f"{z:8.1f}  {dHH:+14.6e}  {Omf:14.6e}  {half_Omf:14.6e}  "
          f"{ratio:10.4f}  {Omm:10.6f}  {verdict}")

print()
print("Note: ratio = (H_GREA/H_LCDM - 1) / (½ Omega_fld)")
print("  ratio ≈ 1  → residual is physical GREA dark-energy tail")
print("  ratio >> 1 → second normalization bug (excess H not explained by rho_fld)")
print()

# Extra: also print rho_fld at each z to verify the fluid itself decays correctly
if fld_rho_key:
    rho_fld_g = np.array(bg_g[fld_rho_key])[np.argsort(np.array(bg_g['z']))]
    rho_crit_g_arr = rho_crit_g
    print(f"{'z':>8}  {'rho_fld':>16}  {'rho_crit':>16}  {'rho_fld/rho_crit':>18}")
    for z in Z_PROBE:
        rf   = float(np.interp(z, z_g, rho_fld_g))
        rc   = float(np.interp(z, z_g, rho_crit_g_arr))
        print(f"{z:8.1f}  {rf:16.6e}  {rc:16.6e}  {rf/rc:18.6e}")

lcdm.struct_cleanup(); lcdm.empty()
grea.struct_cleanup(); grea.empty()
print("\nDone.")
