#!/usr/bin/env python3
"""
Step 1 diagnostic: run GREA and LCDM with default and tighter integration
tolerances.  Print f(z) at z=2,3,5,10 and the GREA-LCDM offsets.
Also print grea_E1 to diagnose whether the w-table uses un/normalised E.
READ-ONLY diagnostic – no CLASS source changes.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from classy import Class
import numpy as np

Z_PROBE = [2.0, 3.0, 5.0, 10.0]

BASE = {
    'h': 0.7,
    'omega_b': 0.022,
    'omega_cdm': 0.12,
    'A_s': 2.1e-9,
    'n_s': 0.965,
    'tau_reio': 0.054,
    'output': '',          # background only
}

LCDM = {**BASE, 'fluid_equation_of_state': 'CLP', 'w0_fld': -1.0, 'wa_fld': 0.0, 'cs2_fld': 1.0}
GREA = {**BASE, 'fluid_equation_of_state': 'GREA', 'sqrt_k_eta0': 3.6, 'cs2_fld': 1.0, 'use_ppf': 'yes'}


def run(params, extra_prec=None):
    c = Class()
    c.set(params)
    if extra_prec:
        c.set(extra_prec)
    c.compute()
    return c


def f_at_z(c, z_list):
    return [c.scale_independent_growth_factor_f(z) for z in z_list]


def print_table(label, f_lcdm, f_grea):
    print(f"\n{'z':>6}  {'f_LCDM':>14}  {'f_GREA':>14}  {'delta_f':>14}  {'delta_f/f_L':>14}")
    for z, fl, fg in zip(Z_PROBE, f_lcdm, f_grea):
        df = fg - fl
        print(f"{z:6.1f}  {fl:14.8f}  {fg:14.8f}  {df:14.8e}  {df/fl:14.8e}")


# ── default precision ────────────────────────────────────────────────────────
print("="*70)
print("DEFAULT precision  (tol_bg=1e-10, a_ini=1e-14, Nloga=40000)")
print("="*70)
lcdm_def = run(LCDM)
grea_def = run(GREA)

fl_def = f_at_z(lcdm_def, Z_PROBE)
fg_def = f_at_z(grea_def, Z_PROBE)
print_table("default", fl_def, fg_def)

# ── tight precision ──────────────────────────────────────────────────────────
TIGHT = {
    'tol_background_integration': 1e-13,
    'a_ini_over_a_today_default': 1e-14,   # same start; tighten only tol
    'background_Nloga': 40000,
}
print("\n" + "="*70)
print("TIGHT precision    (tol_bg=1e-13)")
print("="*70)
lcdm_tight = run(LCDM, TIGHT)
grea_tight = run(GREA, TIGHT)

fl_tight = f_at_z(lcdm_tight, Z_PROBE)
fg_tight = f_at_z(grea_tight, Z_PROBE)
print_table("tight", fl_tight, fg_tight)

# ── tighter start epoch ──────────────────────────────────────────────────────
EARLY = {
    'tol_background_integration': 1e-13,
    'a_ini_over_a_today_default': 1e-16,
    'background_Nloga': 40000,
}
print("\n" + "="*70)
print("EARLY+TIGHT        (tol_bg=1e-13, a_ini=1e-16)")
print("="*70)
lcdm_early = run(LCDM, EARLY)
grea_early = run(GREA, EARLY)

fl_early = f_at_z(lcdm_early, Z_PROBE)
fg_early = f_at_z(grea_early, Z_PROBE)
print_table("early+tight", fl_early, fg_early)

# ── check whether delta converges ───────────────────────────────────────────
print("\n" + "="*70)
print("CONVERGENCE TABLE: delta_f = f_GREA - f_LCDM at z=10")
print("="*70)
for label, fl, fg in [
    ("default    (tol=1e-10, a_ini=1e-14)", fl_def,   fg_def),
    ("tight      (tol=1e-13, a_ini=1e-14)", fl_tight, fg_tight),
    ("early+tight(tol=1e-13, a_ini=1e-16)", fl_early, fg_early),
]:
    df10 = fg[3] - fl[3]   # z=10 is index 3
    print(f"  {label}: delta_f(z=10) = {df10:+.6e}")

# ── grea_E1 diagnostic ───────────────────────────────────────────────────────
print("\n" + "="*70)
print("GREA E(1) diagnostic  (grea_E1 from CLASS struct)")
print("="*70)
try:
    # The CLASS verbose output prints grea_E1; here we compute it manually
    # from the background table: E(z=0) = H(z=0)/H0
    bg = grea_def.get_background()
    z_bg = np.array(bg['z'])
    H_bg = np.array(bg['H [1/Mpc]'])
    # H at z=0
    idx0 = np.argmin(np.abs(z_bg))
    H0_from_bg = H_bg[idx0]
    H0_param = grea_def.Hubble(0)
    print(f"  H0 from bg table:   {H0_from_bg:.10e}  Mpc^-1")
    print(f"  H0 from Hubble(0):  {H0_param:.10e}  Mpc^-1")
    print(f"  h parameter:        {grea_def.h():.10f}")
    # Compare H(z) at several z values to detect any systematic offset
    print("\n  H_GREA/H_LCDM - 1 at probe redshifts:")
    bg_l = lcdm_def.get_background()
    z_l = np.array(bg_l['z']); H_l = np.array(bg_l['H [1/Mpc]'])
    z_g = np.array(bg['z']);   H_g = H_bg
    for z in [0.0, 1.0, 2.0, 5.0, 10.0, 50.0, 200.0]:
        Hl = float(np.interp(z, z_l[::-1], H_l[::-1]))
        Hg = float(np.interp(z, z_g[::-1], H_g[::-1]))
        print(f"    z={z:6.1f}: H_G/H_L - 1 = {Hg/Hl - 1:+.4e}")
except Exception as e:
    print(f"  Could not retrieve background: {e}")

for c in [lcdm_def, grea_def, lcdm_tight, grea_tight, lcdm_early, grea_early]:
    c.struct_cleanup()
    c.empty()

print("\nDone.")
