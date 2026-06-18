#!/usr/bin/env python3
# coding: utf-8

from classy import Class
import numpy as np
import matplotlib.pyplot as plt
from getdist import MCSamples, plots
import argparse
import os
import sys

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(script_dir, '..', 'python')))

OUTPUT_DIR = os.path.abspath(os.path.join(script_dir, '..', 'plots'))
os.makedirs(OUTPUT_DIR, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot GREA CLASS results: power spectrum and optional parameter corner plot."
    )
    parser.add_argument(
        "--output-dir",
        default="plots",
        help="Directory where plot files will be saved.",
    )
    parser.add_argument(
        "--chain",
        default=None,
        help="Optional text or CSV chain file for corner plotting. If omitted, only the power spectrum is plotted.",
    )
    parser.add_argument(
        "--sqrt_k_eta0",
        type=float,
        default=3.6,
        help="GREA curvature parameter g = sqrt(-k) * eta0.",
    )
    parser.add_argument(
        "--cs2_fld",
        type=float,
        default=1.0,
        help="Sound speed squared for the GREA fluid (1 or 0).",
    )
    parser.add_argument(
        "--h",
        type=float,
        default=0.7,
        help="Reduced Hubble constant h.",
    )
    return parser.parse_args()


def build_class_params(args, cs2_fld=1.0):
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    return {
        "output": "mPk tCl pCl lCl",
        "l_max_scalars": 2500,
        "P_k_max_1/Mpc": 5.0,
        "z_pk": 0.0,
        "h": args.h,
        "omega_b": 0.022,
        "omega_cdm": 0.12,
        "A_s": 2.1e-9,
        "n_s": 0.965,
        "tau_reio": 0.054,
        "fluid_equation_of_state": "GREA",
        "sqrt_k_eta0": args.sqrt_k_eta0,
        "cs2_fld": cs2_fld,
        "use_ppf": "yes",
        "base_path": base_path,
    }


def build_lcdm_params(args):
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    omega_b = 0.022
    omega_cdm = 0.12
    A_s = 2.1e-9
    n_s = 0.965
    tau_reio = 0.054
    return {
        "output": "mPk tCl pCl lCl",
        "l_max_scalars": 2500,
        "P_k_max_1/Mpc": 5.0,
        "z_pk": 0.0,
        "h": args.h,
        "omega_b": omega_b,
        "omega_cdm": omega_cdm,
        "A_s": A_s,
        "n_s": n_s,
        "tau_reio": tau_reio,
        "fluid_equation_of_state": "CLP",
        "base_path": base_path,
    }


def plot_pk(models, c_ref, output_dir):
    ks = np.logspace(-3, np.log10(3.0), 200)
    fig, ax = plt.subplots(figsize=(8, 6))
    for label, c in models:
        pk = np.array([c.pk(kval, 0.0) for kval in ks])
        ax.loglog(ks, pk, label=label)
    pk_ref = np.array([c_ref.pk(kval, 0.0) for kval in ks])
    ax.loglog(ks, pk_ref, '--', label='LCDM reference')
    ax.set_xlabel(r'$k \, [\mathrm{Mpc}^{-1}]$')
    ax.set_ylabel(r'$P(k) \, [\mathrm{Mpc}^3]$')
    ax.set_title('GREA linear matter power spectrum')
    ax.legend()
    ax.grid(True, which='both', ls=':')
    path = os.path.join(output_dir, 'grea_pk.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('Saved power-spectrum plot to', path)

    fig, ax = plt.subplots(figsize=(8, 6))
    for label, c in models:
        pk = np.array([c.pk(kval, 0.0) for kval in ks])
        ax.loglog(ks, ks ** 3 * pk / (2.0 * np.pi ** 2), label=label)
    ax.loglog(ks, ks ** 3 * pk_ref / (2.0 * np.pi ** 2),
              '--', label='LCDM reference')
    ax.set_xlabel(r'$k \, [\mathrm{Mpc}^{-1}]$')
    ax.set_ylabel(r'$\Delta^2(k)$')
    ax.set_title('Dimensionless matter power spectrum')
    ax.legend()
    ax.grid(True, which='both', ls=':')
    path = os.path.join(output_dir, 'grea_delta2.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('Saved dimensionless power-spectrum plot to', path)


def plot_cls(models, c_ref, output_dir):
    ell = None
    fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    for label, c in models:
        cls = c.raw_cl(2500)
        ell = np.array(cls['ell'][2:])
        tt = np.array(cls['tt'][2:])
        axs[0].loglog(ell, ell * (ell + 1) * tt /
                      (2.0 * np.pi), label=f'{label} TT')
        if 'pp' in cls:
            lens = np.array(cls['pp'][2:])
            axs[1].loglog(ell, ell * (ell + 1) * lens /
                          (2.0 * np.pi), label=label + ' phi phi')

    cls_ref = c_ref.raw_cl(2500)
    ell_ref = np.array(cls_ref['ell'][2:])
    tt_ref = np.array(cls_ref['tt'][2:])
    axs[0].loglog(ell_ref, ell_ref * (ell_ref + 1) * tt_ref /
                  (2.0 * np.pi), '--', label='LCDM TT')
    if 'pp' in cls_ref:
        lens_ref = np.array(cls_ref['pp'][2:])
        axs[1].loglog(ell_ref, ell_ref * (ell_ref + 1) *
                      lens_ref / (2.0 * np.pi), '--', label='LCDM phi phi')

    axs[0].set_ylabel(r'$\ell(\ell+1)C_\ell^{TT} / 2\pi$')
    axs[0].set_title('GREA TT power spectrum versus LCDM')
    axs[0].legend()
    axs[0].grid(True, which='both', ls=':')

    axs[1].set_xlabel(r'$\ell$')
    axs[1].set_ylabel(r'$\ell(\ell+1)C_\ell^{\phi\phi} / 2\pi$')
    axs[1].set_title('GREA lensing potential power versus LCDM')
    axs[1].legend()
    axs[1].grid(True, which='both', ls=':')

    path = os.path.join(output_dir, 'grea_cls.png')
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('Saved CMB spectrum plot to', path)


def plot_s8(models, c_ref, output_dir):
    labels = []
    s8_values = []
    for label, c in models:
        value = float(c.S8)
        labels.append(label)
        s8_values.append(value)
        print(f'{label}: S8 = {value:.4f}')

    ref_value = float(c_ref.S8)
    labels.append('LCDM ref')
    s8_values.append(ref_value)
    print(f'LCDM ref: S8 = {ref_value:.4f}')

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(labels, s8_values, color=['C0', 'C1', 'C2'][:len(labels)])
    ax.set_ylabel(r'$S_8 = \sigma_8 \sqrt{\Omega_m / 0.3}$')
    ax.set_title('S8 comparison: GREA vs LCDM')
    ax.grid(axis='y', ls=':')
    ax.set_ylim(0, max(s8_values) * 1.2)
    for bar, value in zip(bars, s8_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{value:.3f}', ha='center', va='bottom', fontsize=10)
    path = os.path.join(output_dir, 'grea_s8.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('Saved S8 comparison plot to', path)


def load_chain(chain_path):
    data = np.genfromtxt(chain_path, names=True,
                         delimiter=None, dtype=None, encoding='utf-8')
    if data.dtype.names is None:
        raise ValueError('Chain file must have header names for columns.')
    samples = np.vstack([data[name] for name in data.dtype.names]).T
    return samples, list(data.dtype.names)


def generate_default_chain(args, output_dir):
    np.random.seed(12345)
    npoints = 2000
    omega_b = 0.022
    omega_cdm = 0.12
    A_s = 2.1e-9
    n_s = 0.965
    means = np.array([
        args.sqrt_k_eta0,
        args.h,
        omega_b,
        omega_cdm,
        A_s,
        n_s,
    ])
    sigmas = np.array([0.2, 0.02, 2e-4, 4e-3, 1e-10, 0.004])
    samples = np.random.normal(means, sigmas, size=(npoints, len(means)))
    chain_path = os.path.join(output_dir, 'grea_chain.txt')
    header = 'sqrt_k_eta0 h omega_b omega_cdm A_s n_s'
    np.savetxt(chain_path, samples, header=header, fmt='%.8e')
    print('Saved synthetic chain to', chain_path)
    return chain_path


def plot_corner(chain_path, output_dir):
    path = os.path.join(output_dir, 'grea_corner.png')
    if chain_path is None:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, 'No chain file provided.\nUse --chain <path> to generate a getdist corner plot.',
                ha='center', va='center', fontsize=12)
        ax.axis('off')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print('Saved placeholder corner plot to', path)
        return

    samples, names = load_chain(chain_path)
    labels = [name.replace('_', ' ') for name in names]
    mc = MCSamples(samples=samples, names=names, labels=labels)
    g = plots.get_subplot_plotter()
    g.settings.figure_legend_frame = False
    g.triangle_plot(mc, names, diag1d_kwargs={
                    'alpha': 0.6}, legend_labels=None)
    g.export(path)
    print('Saved corner plot to', path)


def main():
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    grea_models = []
    for cs2 in [1.0, 0.0]:
        label = f'GREA cs={cs2}'
        params = build_class_params(args, cs2_fld=cs2)
        c = Class()
        c.set(params)
        c.compute()
        grea_models.append((label, c))

    if args.cs2_fld not in (0.0, 1.0):
        label = f'GREA cs={args.cs2_fld}'
        params = build_class_params(args, cs2_fld=args.cs2_fld)
        c_custom = Class()
        c_custom.set(params)
        c_custom.compute()
        grea_models.append((label, c_custom))
    else:
        c_custom = None

    params_ref = build_lcdm_params(args)
    c_ref = Class()
    c_ref.set(params_ref)
    c_ref.compute()

    plot_pk(grea_models, c_ref, OUTPUT_DIR)
    plot_cls(grea_models, c_ref, OUTPUT_DIR)
    plot_s8(grea_models, c_ref, OUTPUT_DIR)

    chain_path = args.chain
    if chain_path is None:
        chain_path = generate_default_chain(args, OUTPUT_DIR)
    plot_corner(chain_path, OUTPUT_DIR)

    c_ref.empty()
    for _, c in grea_models:
        c.empty()
    if c_custom is not None:
        c_custom.empty()


if __name__ == '__main__':
    main()
