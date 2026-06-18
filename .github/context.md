# Implementing GREA in CLASS 3.3.4 — Agent Context

This file tells you (the coding agent) **what** to implement and **why**, with
the exact equations and physical hypotheses. The `class_public-3.3.4/` folder is
already present in the workspace.

**Division of labour.** This document is authoritative for the *physics*. It is
*not* authoritative for CLASS internals: every file path, function name, and
struct field below is a strong hint, but you must **open the actual 3.3.4 source
and confirm signatures before editing**. Where the doc says "verify", treat it as
a hard instruction, not a formality. CLASS changes between minor versions.

---

## 0. One-paragraph summary

GREA (General Relativistic Entropic Acceleration) replaces the cosmological
constant with an entropic contribution from the causal horizon. It is governed by
a **single extra parameter** and reduces to a modified Friedmann equation plus a
standard matter growth equation. We implement it as an **effective dark-energy
fluid** inside CLASS: a custom background sector that solves the GREA ODE and
exposes `w(a)`, plus the existing CLASS fluid-perturbation + PPF machinery for the
dark-energy perturbations. The matter perturbations need **no** new physics (see
Hypothesis H1).

---

## 1. Physical hypotheses (the "hypothesis" the model rests on)

- **H1 — Standard matter growth.** GREA lives inside GR. The linear matter
  density contrast obeys the *standard* growth equation with the *standard*
  source `(3/2)·Ω_m,0·δ`; there is **no** dark-energy clustering term and **no**
  modified gravitational coupling `G_eff`. GREA enters growth *only* through the
  modified `a(τ)` in the Hubble-friction term. ⇒ `D(a)`, `f(z)`, `fσ8(z)`, and the
  linear `P(k)` shape are fixed **entirely by the background**.
- **H2 — Effective-fluid treatment (v1).** The entropic component is modelled as
  an imperfect fluid with equation of state `w(a)` (below) and a *prescribed*
  sound speed `c_s²`. The fully consistent perturbed entropic tensor `δf_μν`
  involves a non-local horizon perturbation (`δζ ≠ 0`) and is **out of scope for
  v1**. We adopt `δζ = 0` and **bracket** the unknown by running two limits:
  `c_s² = 1` (smooth) and `c_s² = 0` (clustering). Report both.
- **H3 — Negligible at recombination.** The entropic term ∝ `sinh(2τ)` is
  exponentially small at early times, so GREA is `ΛCDM`-like at recombination.
  ⇒ standard adiabatic initial conditions hold; the **primary** CMB acoustic peaks
  are unchanged. Only the **late-time ISW** (low-ℓ TT) and **CMB lensing**
  `C_ℓ^φφ` are modified — these are the new observables, do not switch them off.
- **H4 — Open universe, curvature in the entropic term.** GREA assumes `k < 0`.
  The curvature does **not** appear as a separate `Ω_k a⁻²` term in `E²(a)`; it is
  encoded inside the entropic term via the comoving volume `V_c`. Do not add a
  standard curvature fluid on top.
- **H5 — `H0` is a normalisation, `H(0)` is derived.** This is the single most
  error-prone point. See §4.

---

## 2. Parameters

Sampled / input:

| symbol | meaning | typical | notes |
|---|---|---|---|
| `g ≡ √(−k)·η0` | curvature scale × conformal time today (dimensionless) | prior `[1, 5]`, best-fit `≈ 3.6` | **the** GREA parameter |
| `Ω_m,0` (or `ω_b`, `ω_cdm`) | matter density | ~0.30 | standard |
| `H0` | normalisation of `τ = H0·η` | — | **not** the physical `H(0)`; see §4 |
| `Ω_r,0` | radiation | fixed | from `T_cmb`, `N_eff` (standard CLASS) |

Derived (compute and expose for cross-checks):

- `α = g / ( E(1)·τ(1) )` — should come out `≈ 1.09` for the best fit (Calderón
  et al. 2025). This is the headline GREA number; use it as a validation anchor.
- `H(0) = H0 · E(1)`  (physical present expansion rate).

**Naming warning.** GREA's `τ ≡ H0·η` is a *rescaled, dimensionless conformal
time*. It is **NOT** CLASS's internal conformal time (also called `tau`). Name the
GREA variable something unambiguous in code (e.g. `tau_grea` / `eta_resc`) to
avoid a silent collision.

---

## 3. Background equations (authoritative)

Define the volume normalisation
```
W ≡ (−k)^{3/2} · V_c = π · [ sinh(2g) − 2g ] ,     g = √(−k)·η0
```

Solve the ODE for the rescaled conformal time `τ(a)` from `a_ini = 1e-11` to
`a = 1`:
```
dτ/da = 1 / [ a² · sqrt( Ω_m,0·a⁻³ + Ω_r,0·a⁻⁴ + (4π/(3a²))·sinh(2τ)/W ) ]
```
with the radiation-era initial condition
```
τ(a_ini) = a_ini / sqrt(Ω_r,0)
```
(The RHS depends on the running value of `τ`; forward integration is stable.)

From the solution:
```
E(a) ≡ H(a)/H0 = sqrt( Ω_m,0·a⁻³ + Ω_r,0·a⁻⁴ + f_G(a) )
f_G(a) ≡ (4π/(3a²)) · sinh(2τ(a)) / W          # GREA "dark energy" density fraction
ρ_GREA(a)/ρ_crit,0 = f_G(a) ,                  ρ_crit,0 = 3H0²/(8πG)
```

Effective equation of state of the entropic component (closed form):
```
w(a) = −(1/3) · [ 2·a·τ'(a)·coth(2τ(a)) + 1 ] ,     τ'(a) = dτ/da
```
Get `dw/da` by finite-differencing `w(a)` on the table (the fluid perturbation
code needs it).

**Self-consistency you get for free:** because `w(a)` is *defined* from
`ρ_GREA(a)` via `w = −1 − (1/3) d ln ρ_GREA / d ln a`, if you feed CLASS this
`w(a)` **and** set the fluid density today to `f_G(1)` (in the normalisation of
§4), CLASS's own fluid-density integral `ρ_fld(a) = ρ_fld,0 · exp(∫_a^1 3(1+w)/a'
da')` will reproduce `ρ_GREA(a)` automatically. So you do **not** need to inject a
tabulated density separately — provide `w(a)` and the correct `Ω_fld,0`. (Verify
how 3.3.4 computes `ρ_fld`: it returns an `integral_fld` from `background_w_fld`;
for a non-analytic `w(a)` you must supply that integral numerically from your
table, **not** the CLP closed form.)

---

## 4. The `H0` vs `H(0)` normalisation — read twice

In GREA, `H0` only sets the scale of `τ`; the ODE for the *shape* `E(a)` depends
only on `{Ω_m,0, Ω_r,0, g}`. The physical present rate is `H(0) = H0·E(1)`, and in
general **`E(1) ≠ 1`**. CLASS, however, hard-assumes `H(a=1) = pba->H0`. Reconcile
them as follows (recommended):

1. Solve the GREA ODE → obtain `E(a)` and `E(1)`.
2. Use the **rescaled** Hubble everywhere: `Ê(a) ≡ E(a)/E(1)`, so `Ê(1) = 1`.
   Set `pba->H0 = ` physical `H(0)`. Then `H(a) = pba->H0 · Ê(a)`.
3. The density fractions CLASS stores are the **physical today** values:
   `Ω_fld,0 = f_G(1)/E(1)²`, `Ω_m,code = Ω_m,0/E(1)²`, `Ω_r,code = Ω_r,0/E(1)²`.
   They close the budget by construction since `Ê(1)² = 1`.
4. Treat the user-facing `Ω_m,0` and `g` as the "GREA-natural" inputs; the
   rescaling by `E(1)²` is internal. Expose physical `Ω_m`, `H(0)`, and `α` as
   derived outputs.

All observable, dimensionless quantities (`D_M/r_d`, `D_H/r_d`, `fσ8`, growth)
depend only on `Ê(a)` (shape) and on `H0·r_d`, so the rescaling is exactly a
choice of normalisation and changes no physics. **If the agent skips this step,
distances and the closure of `Σ Ω_i = 1` will be wrong.** Verify the budget closes
to machine precision in `background_solve` before trusting anything downstream.

---

## 5. Perturbations

- **Matter:** nothing to add (H1). Once the background `Ê(a)` is correct, CLASS's
  standard CDM+baryon perturbations give the right `D(a)`, `fσ8`, `P(k)`.
- **Entropic fluid:** use the existing CLASS fluid perturbation sector with
  - `w(a)`, `dw/da` from §3,
  - `cs2_fld = 1` for run A and `cs2_fld = 0` for run B (the H2 bracket),
  - **`use_ppf = yes` is mandatory.** `w(a)` crosses `−1` (transient phantom
    crossing at `z ≲ 2`); a constant-`c_s²` fluid diverges at the crossing unless
    PPF is on. Verify the PPF path is actually taken for your EoS option.
- **Initial conditions:** GREA is negligible early (H3), so the standard adiabatic
  fld IC apply unchanged. Confirm `ρ_fld/ρ_tot → 0` at `a_ini` in the code.

---

## 6. File-by-file plan (CLASS 3.x layout — **verify each in source**)

CLASS 3.x source lives in `source/`, headers in `include/`. The module order is
`input → background → thermodynamics → perturbations → primordial → fourier →
transfer → harmonic → lensing → output` (note: 3.x uses `fourier.c` and
`harmonic.c`, the former `nonlinear.c`/`spectra.c`).

1. **`include/background.h`** — add a GREA EoS flag to the
   `fluid_equation_of_state` enum (alongside `CLP`, `EDE`); add fields for `g`
   (`sqrt_k_eta0`) and for the precomputed tables (`tau_grea(a)`, `Ê(a)`,
   `w(a)`, `dw/da`, `integral_fld`). Verify the enum name and the struct.

2. **`source/input.c`** — parse `sqrt_k_eta0` (e.g. `class_read_double`); select
   the GREA branch of `fluid_equation_of_state`; route `Omega0_fld` to be set from
   the GREA solution (§4) rather than read directly. Verify the parameter-reading
   macros and the shooting/closure logic CLASS uses for `Omega0_fld`.

3. **`source/background.c`** — the core work:
   - Add `background_grea_integrate(...)`: integrates the §3 ODE on a log-`a`
     grid, fills `tau_grea`, `E`, `f_G`, `w`, `dw/da`, and the cumulative
     `integral_fld`. Call it once during background initialisation, **before** the
     main background table is built.
   - In **`background_w_fld(...)`**: add the GREA branch returning interpolated
     `w(a)`, `dw_over_da`, and the tabulated `integral_fld`. (Confirm this exact
     signature in 3.3.4 — it has historically been
     `(pba, a, *w_fld, *dw_over_da_fld, *integral_fld)`.)
   - Apply the §4 `E(1)` rescaling and set the physical `Ω` budget; assert
     closure.
   - Verify how `ρ_fld` is assembled in `background_functions` and that it picks
     up your `integral_fld` rather than a CLP closed form.

4. **`source/perturbations.c`** — likely **no new equations** needed: ensure the
   fld perturbations read `cs2_fld` and that `use_ppf` engages for the GREA EoS.
   Verify the `delta_fld`/`theta_fld` and PPF code paths execute for your option.

5. **Build & wrapper** — `make clean && make`; rebuild the `classy` Python wrapper
   in `python/`. Expose `sqrt_k_eta0`, derived `alpha`, derived `H(0)`.

Keep all edits guarded by the GREA EoS flag so default CLASS behaviour is
untouched.

---

## 7. Validation (do these in order; do not proceed past a failure)

1. **Background shape.** Plot `Ê(a)` and `w(a)` vs `z`. Check: transient phantom
   crossing at `z ≲ 2`; `w0 ≡ w(0) ≈ −1`; slope `wa ≡ dw/da|_{a=1} ≈ −0.3`;
   derived `α ≈ 1.09` for the best-fit input. Cross-check against `greapy`
   (`github.com/rcalderonb6/greapy`) if available, and Fig. 1 of Calderón et al.
   2025.
2. **Closure.** `Σ Ω_i = 1` and `H(a=1) = pba->H0` to machine precision.
3. **Growth.** `fσ8(z)` and the growth index `γ(z) ≡ ln f / ln Ω_m(a)`. Check
   `γ(0) ≈ 0.55` but `dγ/dz > 0` (opposite sign to ΛCDM — a GREA signature).
   These must come out right *automatically* from the background (H1); if they
   don't, the background is wrong, not the perturbations.
4. **CMB.** Confirm primary TT peaks ≈ ΛCDM (H3), but a modified **low-ℓ ISW** and
   a modified `C_ℓ^φφ`. The `cs2 = 1` vs `cs2 = 0` runs should differ **only** in
   ISW/lensing, not in the peaks or in `P(k)` — that difference *is* the H2
   systematic.

---

## 8. Out of scope for v1 / open issues (do not try to solve these now)

- The non-local `δζ ≠ 0` horizon perturbation (the fully consistent `δf_μν`).
  v1 uses `δζ = 0` + the `c_s²` bracket. Leave a clear `TODO(deltazeta)` marker.
- Non-linear `P(k)` (Halofit/HMcode are ΛCDM-calibrated; do not trust them for
  GREA). Linear only for v1.
- A standalone curvature fluid — forbidden by H4.

---

## 9. Reference equations, copy-paste pseudocode

```python
# background precomputation (schematic; port to C)
import numpy as np
from scipy.integrate import solve_ivp

def grea_background(g, Om, Or, a_ini=1e-11, n=4000):
    W = np.pi * (np.sinh(2*g) - 2*g)
    def rhs(a, tau):
        inside = Om*a**-3 + Or*a**-4 + (4*np.pi/(3*a**2))*np.sinh(2*tau[0])/W
        return [1.0 / (a**2 * np.sqrt(inside))]
    a_grid = np.logspace(np.log10(a_ini), 0.0, n)
    sol = solve_ivp(rhs, [a_ini, 1.0], [a_ini/np.sqrt(Or)],
                    t_eval=a_grid, rtol=1e-9, atol=1e-12)
    a   = sol.t
    tau = sol.y[0]
    fG  = (4*np.pi/(3*a**2))*np.sinh(2*tau)/W
    E   = np.sqrt(Om*a**-3 + Or*a**-4 + fG)
    taup = 1.0/(a**2 * E)                       # dτ/da = 1/(a² E)
    w   = -(1.0/3.0)*(2*a*taup/np.tanh(2*tau) + 1.0)   # coth = 1/tanh
    E1  = E[-1]
    alpha = g/(E1*tau[-1])
    return dict(a=a, tau=tau, E=E, E1=E1, w=w, fG=fG, alpha=alpha)
```

Use this exact routine to generate ground-truth tables and diff the C
implementation against it during validation.

---

## 10. Key references (already in the project `.bib`)

- García-Bellido 2024, *Dark Energy predictions from GREA: Background and linear
  perturbation theory*, arXiv:2405.02895 — background ODE, growth eq, fσ8, ISW.
- Calderón, García-Bellido et al. 2025, arXiv:2509.21491 — Eqs. (2.1)–(5.3) used
  above; greapy; best-fit `α ≈ 1.09`, `w0 ≈ −1`, `wa ≈ −0.3`.
- Graziotti, De Leo, Martinelli 2026, arXiv:2603.01934 — independent background
  implementation, ODE in the same form, `H0` vs `H(0)` discussion.
- Gagnon & Lesgourgues 2011, arXiv:1107.1503 — bulk-viscosity perturbation
  formalism (needed only when you eventually attempt the `δζ` derivation).
