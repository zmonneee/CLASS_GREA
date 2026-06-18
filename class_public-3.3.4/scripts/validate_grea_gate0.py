#!/usr/bin/env python3
# coding: utf-8

from classy import Class
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(script_dir, '..', 'python')))

OUTPUT_DIR = os.path.abspath(os.path.join(script_dir, '..', 'plots'))
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Fixed reference cosmology for all runs
COMMON_PARAMS = {
    'output': 'mPk tCl pCl lCl',
    'l_max_scalars': 2500,
    'P_k_max_1/Mpc': 5.0,
    'z_pk': 0.0,
    'h': 0.7,
    'omega_b': 0.022,
    'omega_cdm': 0.12,
    'A_s': 2.1e-9,
    'n_s': 0.965,
    'tau_reio': 0.054,
    'base_path': os.path.abspath(os.path.join(script_dir, '..')),
}


LCDM_PARAMS = {
    **COMMON_PARAMS,
    'fluid_equation_of_state': 'CLP',
    'w0_fld': -1.0,
    'wa_fld': 0.0,
    'cs2_fld': 1.0,
}


def grea_params(cs2_fld):
    return {
        **COMMON_PARAMS,
        'fluid_equation_of_state': 'GREA',
        'sqrt_k_eta0': 3.6,
        'cs2_fld': cs2_fld,
        'use_ppf': 'yes',
    }


def build_class(params):
    c = Class()
    c.set(params)
    c.compute()
    return c


def sort_by_z(z, y):
    order = np.argsort(z)
    return z[order], y[order]


def get_background_arrays(c):
    bg = c.get_background()
    if 'H [1/Mpc]' not in bg:
        raise RuntimeError(
            "Background output does not contain 'H [1/Mpc]' key")
    if 'z' not in bg:
        raise RuntimeError("Background output does not contain 'z' key")
    if 'Omega_m(z)' not in bg:
        raise RuntimeError(
            "Background output does not contain 'Omega_m(z)' key")
    z = np.array(bg['z'], dtype=float)
    H = np.array(bg['H [1/Mpc]'], dtype=float)
    Omega_m = np.array(bg['Omega_m(z)'], dtype=float)
    z, H = sort_by_z(z, H)
    z, Omega_m = sort_by_z(z, Omega_m)
    return z, H, Omega_m


def interpolate_on_grid(z, y, z_grid):
    return np.interp(z_grid, z, y)


def get_gamma(z_grid, f_grid, Omega_m_grid):
    if np.any(Omega_m_grid <= 0):
        raise ValueError(
            'Omega_m(z) is non-positive at some z values, cannot compute gamma')
    return np.log(f_grid) / np.log(Omega_m_grid)


def compute_f(c, z_values):
    if not hasattr(c, 'scale_independent_growth_factor_f'):
        raise AttributeError(
            'scale_independent_growth_factor_f not found in classy wrapper; check CLASS 3.3.4 API')
    try:
        return np.array([float(c.scale_independent_growth_factor_f(z)) for z in z_values], dtype=float)
    except Exception as e:
        raise RuntimeError(
            f'Failed to evaluate scale_independent_growth_factor_f: {e}')


def gate0_background_noop(lcdm, grea):
    z_lcdm, H_lcdm, _ = get_background_arrays(lcdm)
    z_grea, H_grea, _ = get_background_arrays(grea)

    z_plot = np.unique(np.concatenate([
        z_lcdm[(z_lcdm > 1e-3) & (z_lcdm <= 5.0)],
        z_grea[(z_grea > 1e-3) & (z_grea <= 5.0)],
        np.logspace(-3, np.log10(5.0), 1000),
    ]))
    if not np.any(z_plot):
        raise RuntimeError('No z values available in the 1e-3<z<=5.0 range')

    H_lcdm_plot = interpolate_on_grid(z_lcdm, H_lcdm, z_plot)
    H_grea_plot = interpolate_on_grid(z_grea, H_grea, z_plot)
    ratio_plot = H_grea_plot / H_lcdm_plot - 1.0
    gate0 = np.max(np.abs(ratio_plot))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(z_plot, ratio_plot, lw=1.5)
    ax.set_xscale('log')
    ax.set_xlim(1e-3, 5.0)
    ax.set_xlabel('z')
    ax.set_ylabel(r'$H_{GREA}(z)/H_{LCDM}(z) - 1$')
    ax.set_title('Gate 0: GREA vs LCDM background H(z) ratio')
    ax.grid(True, ls=':')
    ax.set_ylim(np.min(ratio_plot) * 1.1, np.max(ratio_plot) * 1.1)
    path = os.path.join(OUTPUT_DIR, 'gate0_background_ratio.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Gate 0 plot saved to {path}')
    return gate0, z_plot, ratio_plot


def compute_gamma_slope(c, tag):
    z_grid = np.linspace(0.0, 5.0, 25)
    f_grid = compute_f(c, z_grid)
    bg = c.get_background()
    z_bg = np.array(bg['z'], dtype=float)
    Omega_m_bg = np.array(bg['Omega_m(z)'], dtype=float)
    z_bg, Omega_m_bg = sort_by_z(z_bg, Omega_m_bg)
    Omega_m_grid = interpolate_on_grid(z_bg, Omega_m_bg, z_grid)
    gamma_grid = get_gamma(z_grid, f_grid, Omega_m_grid)
    slope = np.polyfit(z_grid, gamma_grid, 1)[0]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(z_grid, gamma_grid, marker='o', ms=4)
    ax.set_xlabel('z')
    ax.set_ylabel('gamma(z) = ln f / ln Omega_m')
    ax.set_title('Growth index gamma(z)')
    ax.grid(True, ls=':')
    safe_tag = tag.replace(' ', '_').replace('=', '').replace('.', 'p')
    path = os.path.join(OUTPUT_DIR, f'gamma_vs_z_{safe_tag}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Gamma plot saved to {path}')
    return slope, z_grid, gamma_grid


def save_s8_table(results):
    path = os.path.join(OUTPUT_DIR, 'gate0_s8_table.txt')
    with open(path, 'w') as f:
        f.write('model,S8\n')
        for label, value in results:
            f.write(f'{label},{value:.6f}\n')
    print(f'S8 table saved to {path}')


def main():
    print('Running Gate 0 diagnostics for GREA CLASS build...')
    lcdm = build_class(LCDM_PARAMS)
    grea_cs1 = build_class(grea_params(1.0))
    grea_cs0 = build_class(grea_params(0.0))

    gate0_value, z_grid, ratio = gate0_background_noop(lcdm, grea_cs1)

    s8_results = [
        ('LCDM', float(lcdm.S8)),
        ('GREA cs=1.0', float(grea_cs1.S8)),
        ('GREA cs=0.0', float(grea_cs0.S8)),
    ]
    save_s8_table(s8_results)
    print('S8 values:')
    for label, value in s8_results:
        print(f'  {label}: {value:.6f}')

    slope_cs1, z_gamma, gamma_cs1 = compute_gamma_slope(grea_cs1, 'cs1')
    slope_cs0, _, gamma_cs0 = compute_gamma_slope(grea_cs0, 'cs0')
    positive_cs1 = slope_cs1 > 0
    positive_cs0 = slope_cs0 > 0

    print('\nGate 0 results:')
    print(f'  max|H_GREA/H_LCDM - 1| = {gate0_value:.3e}')
    print(
        f'  gamma_slope(cs=1.0) = {slope_cs1:.4e} -> positive={positive_cs1}')
    print(
        f'  gamma_slope(cs=0.0) = {slope_cs0:.4e} -> positive={positive_cs0}')

    with open(os.path.join(OUTPUT_DIR, 'gate0_summary.txt'), 'w') as f:
        f.write(f'Gate 0 number: {gate0_value:.6e}\n')
        f.write('S8 table:\n')
        for label, value in s8_results:
            f.write(f'{label}: {value:.6f}\n')
        f.write(
            f'gamma_slope(cs=1.0): {slope_cs1:.6e} positive={positive_cs1}\n')
        f.write(
            f'gamma_slope(cs=0.0): {slope_cs0:.6e} positive={positive_cs0}\n')

    lcdm.empty()
    grea_cs1.empty()
    grea_cs0.empty()
    print('\nGate 0 diagnostics completed.')


if __name__ == '__main__':
    main()
