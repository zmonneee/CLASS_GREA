#!/usr/bin/env python3
"""
validate_grea.py — Regression test suite for GREA-patched CLASS 3.3.4.

Runs both a LCDM reference and a GREA model, checks nine gates with fixed
tolerances, and exits non-zero if any gate fails.

Usage:
    python validate_grea.py            # run all gates
    python validate_grea.py --reset    # overwrite the stored LCDM reference

Binary freshness guard is MANDATORY: if the compiled .so predates
background.c or perturbations.c the script exits non-zero immediately.
"""

import sys
import os
import json
import argparse
import glob
import datetime
import numpy as np

# ── path setup ─────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT       = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
sys.path.insert(0, ROOT)

from classy import Class

LCDM_REF_FILE  = os.path.join(SCRIPT_DIR, 'lcdm_reference.json')
DIAG_DIR       = os.path.join(SCRIPT_DIR, '..', 'plots')
os.makedirs(DIAG_DIR, exist_ok=True)

_C_KMS = 2.99792458e5   # speed of light in km/s

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 0 — BINARY FRESHNESS GUARD
# ═══════════════════════════════════════════════════════════════════════════════

def _mtime(path):
    return os.path.getmtime(path) if os.path.exists(path) else 0.0

def check_binary_freshness():
    so_pattern = os.path.join(ROOT, '_classy*.so')
    matches = sorted(glob.glob(so_pattern))
    if not matches:
        print("FATAL: no _classy*.so found under", ROOT)
        sys.exit(1)

    so_path = matches[0]
    so_mtime = _mtime(so_path)

    src_bg   = os.path.join(ROOT, 'source', 'background.c')
    src_pt   = os.path.join(ROOT, 'source', 'perturbations.c')

    def fmt(t):
        return datetime.datetime.fromtimestamp(t).strftime('%Y-%m-%d %H:%M:%S')

    print("=" * 70)
    print("BINARY FRESHNESS CHECK")
    print("=" * 70)
    print(f"  _classy.so      : {fmt(so_mtime)}  ({os.path.basename(so_path)})")
    print(f"  background.c    : {fmt(_mtime(src_bg))}")
    print(f"  perturbations.c : {fmt(_mtime(src_pt))}")

    stale = False
    if so_mtime < _mtime(src_bg):
        print(f"\n  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"  !!! STALE BINARY — .so is older than background.c             !!!")
        print(f"  !!! Rebuild: make libclass.a && python setup.py build_ext --inplace !!!")
        print(f"  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        stale = True
    if so_mtime < _mtime(src_pt):
        print(f"\n  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"  !!! STALE BINARY — .so is older than perturbations.c          !!!")
        print(f"  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        stale = True
    if not stale:
        print(f"  FRESH ✓  .so is newer than both source files.")

    if stale:
        sys.exit(1)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — BUILD BOTH CLASS INSTANCES
# ═══════════════════════════════════════════════════════════════════════════════

COMMON = {
    'omega_b':        0.02237,
    'omega_cdm':      0.1200,
    'h':              0.70,
    'A_s':            2.1e-9,
    'n_s':            0.9649,
    'tau_reio':       0.0544,
    'output':         'tCl pCl lCl mPk',
    'lensing':        'yes',
    'P_k_max_1/Mpc':  5.0,
    'z_max_pk':       3.0,
    'l_max_scalars':  2500,
    'base_path':      ROOT,
}

LCDM_EXTRA = {
    # CLP fluid with w0=-1, wa=0 identical to Λ.
    # Omega_Lambda NOT specified: CLASS auto-fills it from the budget equation
    # (background.c:3522). This gives has_lambda=TRUE, has_fld=FALSE.
    # The dark energy then appears as rho_lambda in the background table.
    'fluid_equation_of_state': 'CLP',
    'w0_fld':  -1.0,
    'wa_fld':   0.0,
    'cs2_fld':  1.0,
}

GREA_EXTRA = {
    # Parameter names verified against source/input.c lines 3614-3624.
    # input.c:3588 matches "GREA" or "grea" in fluid_equation_of_state.
    # input.c:3619 reads sqrt_k_eta0; input.c:3618 reads cs2_fld.
    # input.c:3623 forces use_ppf=TRUE for GREA regardless of input,
    # but we set it explicitly for clarity.
    'fluid_equation_of_state': 'GREA',
    'sqrt_k_eta0': 3.6,
    'cs2_fld':     1.0,
    'use_ppf':     'yes',
}

def build(extra):
    c = Class()
    c.set({**COMMON, **extra})
    c.compute()
    return c

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — GATE INFRASTRUCTURE
# ═══════════════════════════════════════════════════════════════════════════════

_gate_results = []   # list of dicts

def gate(name, measured, expected_str, passed, detail=''):
    """Record one gate result and print it immediately."""
    status = 'PASS ✓' if passed else 'FAIL ✗'
    _gate_results.append(dict(name=name, measured=measured,
                              expected=expected_str, passed=passed))
    meas_str = f'{measured:.6g}' if isinstance(measured, float) else str(measured)
    print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    print(f"       measured  : {meas_str}")
    print(f"       expected  : {expected_str}")
    if detail:
        print(f"       detail    : {detail}")

def _sorted(bg, *keys):
    """Sort background arrays by z ascending."""
    z = np.array(bg['z']); idx = np.argsort(z)
    result = [z[idx]]
    for k in keys:
        result.append(np.array(bg[k])[idx])
    return tuple(result)

def _interp(z_target, z_arr, y_arr):
    return float(np.interp(z_target, z_arr, y_arr))

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — GATE IMPLEMENTATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def gate1_H0_shared(lcdm, grea):
    """Gate 1: H(0) for both models equals 70.000 km/s/Mpc."""
    print("\n── Gate 1: H(0) shared ─────────────────────────────────────")
    H0_lcdm = lcdm.h() * 100.0
    H0_grea  = grea.h() * 100.0
    dH = abs(H0_grea - H0_lcdm)
    tol = 1e-5
    gate('G1 |H0_GREA - H0_LCDM|', dH, f'< {tol} km/s/Mpc', dH < tol,
         f'H0_LCDM={H0_lcdm:.9f}  H0_GREA={H0_grea:.9f} km/s/Mpc')


def gate2_sound_horizon(lcdm, grea):
    """Gate 2: Sound horizon at recombination preserved."""
    print("\n── Gate 2: r_s(rec) preserved ──────────────────────────────")
    d_l = lcdm.get_current_derived_parameters(['rs_rec'])
    d_g = grea.get_current_derived_parameters(['rs_rec'])
    rs_l = d_l['rs_rec']   # Mpc, from thermodynamics struct th.rs_rec
    rs_g = d_g['rs_rec']
    frac = abs(rs_g / rs_l - 1.0)
    tol  = 1e-4
    gate('G2 |Δr_s/r_s|', frac, f'< {tol}', frac < tol,
         f'rs_rec_LCDM={rs_l:.6f} Mpc  rs_rec_GREA={rs_g:.6f} Mpc')


def gate3_omega_densities(grea):
    """Gate 3: Internal omega_b and omega_cdm of GREA run match inputs."""
    print("\n── Gate 3: Matter densities preserved ───────────────────────")
    omega_b_got   = grea.omega_b()            # = Omega0_b * h²  (classy.pyx:153)
    omega_cdm_got = grea.Omega_cdm * grea.h()**2  # no direct omega_cdm method
    tol = 1e-5
    db  = abs(omega_b_got   - 0.02237)
    dc  = abs(omega_cdm_got - 0.1200)
    gate('G3a |Δomega_b|',   db, f'< {tol}', db < tol,
         f'got={omega_b_got:.6f}  input=0.02237')
    gate('G3b |Δomega_cdm|', dc, f'< {tol}', dc < tol,
         f'got={omega_cdm_got:.6f}  input=0.12000')


def gate4_early_negligible(bg_l, bg_g):
    """Gate 4: GREA negligible at z=1000 (H ratio < 1e-4)."""
    print("\n── Gate 4: GREA negligible at early times ───────────────────")
    z_l, H_l = _sorted(bg_l, 'H [1/Mpc]')
    z_g, H_g = _sorted(bg_g, 'H [1/Mpc]')
    Hl_1000 = _interp(1000., z_l, H_l)
    Hg_1000 = _interp(1000., z_g, H_g)
    ratio   = abs(Hg_1000 / Hl_1000 - 1.0)
    tol     = 1e-4
    gate('G4 |H_GREA/H_LCDM - 1| at z=1000', ratio, f'< {tol}', ratio < tol)


def gate5_physical_tail(bg_l, bg_g):
    """Gate 5: H residual = ½·(Ω_fld_GREA − Ω_fld_LCDM) at z=10,50,200,1000."""
    print("\n── Gate 5: High-z H residual is physical entropic tail ──────")

    # Background arrays
    z_l, H_l, rc_l = _sorted(bg_l, 'H [1/Mpc]', '(.)rho_crit')
    z_g, H_g, rc_g = _sorted(bg_g, 'H [1/Mpc]', '(.)rho_crit')

    # GREA dark energy: rho_fld (verified key from live probe)
    _, rfl_g = _sorted(bg_g, '(.)rho_fld')

    # LCDM dark energy: background.c sets has_lambda=TRUE for auto-fill CLP.
    # The CLP with auto-Omega_Lambda puts DE in rho_lambda, not rho_fld.
    lcdm_keys = list(bg_l.keys())
    if '(.)rho_lambda' in lcdm_keys:
        _, rfl_l = _sorted(bg_l, '(.)rho_lambda')
        lcdm_de_key = '(.)rho_lambda'
    elif '(.)rho_fld' in lcdm_keys:
        _, rfl_l = _sorted(bg_l, '(.)rho_fld')
        lcdm_de_key = '(.)rho_fld'
    else:
        print("  ERROR: cannot find LCDM DE density key in background; "
              f"available keys: {lcdm_keys}")
        gate('G5 physical tail (z=10)', float('nan'), '|ratio-1| < 0.01', False,
             'LCDM DE key not found')
        return

    tol = 0.01    # ratio within 1% of 1.0
    all_ok = True
    for z in [10.0, 50.0, 200.0, 1000.0]:
        Hl  = _interp(z, z_l, H_l);  Hg  = _interp(z, z_g, H_g)
        rcl = _interp(z, z_l, rc_l); rcg = _interp(z, z_g, rc_g)
        rfl = _interp(z, z_l, rfl_l); rfg = _interp(z, z_g, rfl_g)
        Omf_G = rfg / rcg;  Omf_L = rfl / rcl
        dOmf  = Omf_G - Omf_L
        dHH   = Hg / Hl - 1.0
        if abs(dOmf) < 1e-30:
            ratio = float('nan'); ok = False
        else:
            ratio = dHH / (0.5 * dOmf); ok = abs(ratio - 1.0) < tol
        all_ok = all_ok and ok
        gate(f'G5 ratio at z={z:.0f}', ratio, f'|ratio-1| < {tol}', ok,
             f'dH/H={dHH:+.4e}  ½dOmf={0.5*dOmf:+.4e}  key_LCDM={lcdm_de_key}')


def gate6_dip_survives(bg_l, bg_g):
    """Gate 6: GREA Gate-0 dip in H(z) survives (1.5–2.5% near z≈1)."""
    print("\n── Gate 6: Gate-0 dip survives ──────────────────────────────")
    z_l, H_l = _sorted(bg_l, 'H [1/Mpc]')
    z_g, H_g = _sorted(bg_g, 'H [1/Mpc]')

    # Restrict to 0 < z < 3 (physical DE era)
    mask_l = (z_l > 0.) & (z_l < 3.)
    mask_g = (z_g > 0.) & (z_g < 3.)
    z_probe = np.union1d(z_l[mask_l], z_g[mask_g])
    Hl_probe = np.interp(z_probe, z_l, H_l)
    Hg_probe = np.interp(z_probe, z_g, H_g)
    ratio    = Hg_probe / Hl_probe - 1.0

    idx_max = np.argmax(np.abs(ratio))
    peak_z  = float(z_probe[idx_max])
    peak_v  = float(ratio[idx_max])     # typically negative (GREA dip)
    peak_abs = abs(peak_v)

    dip_ok  = (0.015 <= peak_abs <= 0.025)
    z_ok    = (0.5   <= peak_z   <= 2.0)
    ok      = dip_ok and z_ok
    gate('G6 dip magnitude', peak_abs,
         '0.015 ≤ |dip| ≤ 0.025', dip_ok,
         f'max|H_G/H_L-1|={peak_abs:.4f} at z={peak_z:.3f}')
    gate('G6 dip redshift', peak_z,
         '0.5 ≤ z_peak ≤ 2.0', z_ok,
         f'z_peak={peak_z:.3f}')


def gate7_w_trajectory(bg_g):
    """Gate 7: w(a) trajectory — w0, wa, phantom crossings."""
    print("\n── Gate 7: w(a) trajectory ──────────────────────────────────")

    # Extract w(z) from background table
    z_w, w_fld = _sorted(bg_g, '(.)w_fld')
    a_w = 1.0 / (1.0 + z_w)       # a increases as z decreases

    # w0 = w(z=0)
    w0 = _interp(0.0, z_w, w_fld)

    # wa = -dw/da |_{a=1}
    # Use polynomial fit through points with a > 0.90 (z < 0.111).
    # Background has 40000 log-spaced points so ~1000 points in this range.
    mask = (a_w >= 0.90) & (a_w <= 1.0)
    a_near = a_w[mask]; w_near = w_fld[mask]
    if len(a_near) >= 4:
        # Quadratic fit; derivative is linear: dw/da = 2c0*a + c1 at a=1
        c = np.polyfit(a_near, w_near, deg=2)
        dw_da_at1 = 2.0 * c[0] * 1.0 + c[1]
    else:
        # Fallback: linear two-point from z=0 and z≈0.1
        w_a90 = _interp(1.0/0.90 - 1.0, z_w, w_fld)
        dw_da_at1 = (w0 - w_a90) / (1.0 - 0.90)
    wa = -dw_da_at1

    # Phantom crossings: sign changes of (w+1)
    w_plus1 = w_fld + 1.0
    signs   = np.sign(w_plus1)
    sc_idx  = np.where(np.diff(signs) != 0)[0]
    crossings = []
    for i in sc_idx:
        z0, z1 = z_w[i], z_w[i + 1]
        p0, p1 = w_plus1[i], w_plus1[i + 1]
        z_cross = z0 + (0.0 - p0) * (z1 - z0) / (p1 - p0)
        crossings.append(float(z_cross))

    # Gate checks
    w0_tol  = 0.02
    wa_lo   = -0.40;  wa_hi = -0.25
    ok_w0   = abs(w0 + 1.0) < w0_tol
    ok_wa   = wa_lo <= wa <= wa_hi
    ok_cross = len(crossings) >= 2  # transient: two crossings expected

    gate('G7a w0 = w(z=0)', w0,
         f'|w0+1| < {w0_tol}  (≈-1.0)', ok_w0)
    gate('G7b wa = -dw/da|_{{a=1}}', wa,
         f'{wa_lo} ≤ wa ≤ {wa_hi}  (≈-0.3)', ok_wa,
         f'computed via poly fit of w(a) for a∈[0.90,1.0]')
    gate('G7c phantom crossings', len(crossings),
         '≥ 2 (transient phantom: two sign changes of w+1)', ok_cross,
         f'crossings at z ≈ ' + ', '.join(f'{zc:.4f}' for zc in sorted(crossings)))

    return crossings, w0, wa


def gate8_growth_direction(lcdm, grea):
    """Gate 8: Record sigma8/S8 and verify GREA enhances growth (no threshold)."""
    print("\n── Gate 8: Growth direction (record-only) ───────────────────")
    s8_l = lcdm.sigma8(); s8_g = grea.sigma8()
    S8_l = lcdm.S8();     S8_g = grea.S8()
    Om_l = lcdm.Omega_m();Om_g = grea.Omega_m()
    ok = s8_g > s8_l   # expected: GREA enhances growth at fixed A_s
    gate('G8 sigma8_GREA > sigma8_LCDM', float(s8_g - s8_l),
         '> 0 (GREA enhances growth)', ok,
         f'sigma8_LCDM={s8_l:.6f}  sigma8_GREA={s8_g:.6f}  '
         f'S8_LCDM={S8_l:.6f}  S8_GREA={S8_g:.6f}  '
         f'Omega_m_LCDM={Om_l:.6f}  Omega_m_GREA={Om_g:.6f}')
    return dict(sigma8_lcdm=float(s8_l), sigma8_grea=float(s8_g),
                S8_lcdm=float(S8_l),     S8_grea=float(S8_g))


def gate9_lcdm_unchanged(lcdm, sigma8_lcdm, reset=False):
    """Gate 9: LCDM instance unchanged vs stored reference (anti-coupling guard)."""
    print("\n── Gate 9: LCDM unchanged guard ─────────────────────────────")

    rs_drag_l = lcdm.rs_drag()
    H0_l      = lcdm.h() * 100.0
    s8_l      = sigma8_lcdm

    current = dict(H0=H0_l, rs_drag=rs_drag_l, sigma8=s8_l)

    if reset or not os.path.exists(LCDM_REF_FILE):
        with open(LCDM_REF_FILE, 'w') as f:
            json.dump(current, f, indent=2)
        action = 'reset' if reset else 'first run'
        gate('G9 LCDM reference written', 0.0,
             f'N/A ({action} — wrote reference to lcdm_reference.json)', True,
             f'H0={H0_l:.9f}  rs_drag={rs_drag_l:.9f}  sigma8={s8_l:.9f}')
        return

    with open(LCDM_REF_FILE) as f:
        ref = json.load(f)

    tol = 1e-6
    drifts = {k: abs(current[k] - ref[k]) for k in current}
    all_ok = all(d < tol for d in drifts.values())

    gate('G9 LCDM H0 stable',     drifts['H0'],     f'< {tol}', drifts['H0']     < tol,
         f'current={current["H0"]:.9f}  ref={ref["H0"]:.9f}')
    gate('G9 LCDM rs_drag stable', drifts['rs_drag'], f'< {tol}', drifts['rs_drag'] < tol,
         f'current={current["rs_drag"]:.9f}  ref={ref["rs_drag"]:.9f}')
    gate('G9 LCDM sigma8 stable', drifts['sigma8'],  f'< {tol}', drifts['sigma8']  < tol,
         f'current={current["sigma8"]:.9f}  ref={ref["sigma8"]:.9f}')

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — DIAGNOSTIC ARRAY SAVE (versioned, never overwrite)
# ═══════════════════════════════════════════════════════════════════════════════

def save_diagnostics(bg_l, bg_g, tag):
    """Save background arrays to a versioned text file."""
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(DIAG_DIR, f'validate_grea_bg_{ts}_{tag}.txt')

    z_l, H_l = _sorted(bg_l, 'H [1/Mpc]')
    z_g, H_g, w_g = _sorted(bg_g, 'H [1/Mpc]', '(.)w_fld')
    z_probe = np.union1d(z_l, z_g)
    Hl_p = np.interp(z_probe, z_l, H_l)
    Hg_p = np.interp(z_probe, z_g, H_g)
    wg_p = np.interp(z_probe, z_g, w_g)

    header = ('z  H_LCDM[1/Mpc]  H_GREA[1/Mpc]  H_ratio-1  w_GREA')
    data   = np.column_stack([z_probe, Hl_p, Hg_p, Hg_p/Hl_p - 1., wg_p])
    np.savetxt(path, data, header=header, fmt='%.9e')
    print(f"\n  Diagnostic arrays saved to {os.path.basename(path)}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — SUMMARY TABLE AND EXIT
# ═══════════════════════════════════════════════════════════════════════════════

def print_summary():
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"  {'Gate':<35}  {'Measured':>14}  {'Expected/tol':<22}  {'Result'}")
    print(f"  {'-'*35}  {'-'*14}  {'-'*22}  {'-'*6}")
    n_fail = 0
    for r in _gate_results:
        meas = r['measured']
        meas_str = f'{meas:.5g}' if isinstance(meas, float) else str(meas)
        status   = 'PASS ✓' if r['passed'] else 'FAIL ✗'
        if not r['passed']:
            n_fail += 1
        print(f"  {r['name']:<35}  {meas_str:>14}  {r['expected']:<22}  {status}")
    print("=" * 78)
    if n_fail == 0:
        print(f"  ALL GATES PASSED ({len(_gate_results)} checks)")
    else:
        print(f"  {n_fail} GATE(S) FAILED — see output above for details")
    print("=" * 78)
    return n_fail

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reset', action='store_true',
                        help='Overwrite stored LCDM reference values (Gate 9)')
    args = parser.parse_args()

    # ── Step 0: freshness guard ─────────────────────────────────────────────
    check_binary_freshness()

    # ── Step 1: build ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("BUILDING CLASS INSTANCES")
    print("=" * 70)
    print("  Building LCDM (CLP w0=-1, wa=0) ...")
    lcdm = build(LCDM_EXTRA)
    print("  Building GREA (sqrt_k_eta0=3.6) ...")
    grea = build(GREA_EXTRA)
    print("  Done.")

    # ── Step 2: extract background tables (one call each, reused by all gates)
    bg_l = lcdm.get_background()
    bg_g = grea.get_background()

    # Verify key presence — fail loudly rather than silently using a wrong field
    required_grea_keys = {'H [1/Mpc]', '(.)rho_fld', '(.)w_fld', '(.)rho_crit'}
    missing = required_grea_keys - set(bg_g.keys())
    if missing:
        print(f"\nFATAL: GREA background missing expected keys: {missing}")
        print(f"  Available keys: {list(bg_g.keys())}")
        sys.exit(1)

    # ── Step 3: run gates ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("GATE CHECKS")
    print("=" * 70)

    gate1_H0_shared(lcdm, grea)
    gate2_sound_horizon(lcdm, grea)
    gate3_omega_densities(grea)
    gate4_early_negligible(bg_l, bg_g)
    gate5_physical_tail(bg_l, bg_g)
    gate6_dip_survives(bg_l, bg_g)
    crossings, w0, wa = gate7_w_trajectory(bg_g)
    g8 = gate8_growth_direction(lcdm, grea)
    gate9_lcdm_unchanged(lcdm, g8['sigma8_lcdm'], reset=args.reset)

    # ── Step 4: save diagnostic arrays ─────────────────────────────────────
    save_diagnostics(bg_l, bg_g, tag='validate')

    # ── Step 5: cleanup ─────────────────────────────────────────────────────
    lcdm.struct_cleanup(); lcdm.empty()
    grea.struct_cleanup(); grea.empty()

    # ── Step 6: summary and exit code ───────────────────────────────────────
    n_fail = print_summary()
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == '__main__':
    main()
