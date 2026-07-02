#!/usr/bin/env python
import os
from cobaya.yaml import yaml_load_file, yaml_dump

SRC_DIR = "/users/msq26sd/GREA/CLASS_yamls"
PROJ    = "/mnt/parscratch/users/msq26sd/GREA"

# source CCG CLASS yaml -> (GREA folder, GREA yaml name)
RUNS = [
    ("CMB-SPA.yaml",          "grea_Run2_CMB-SPA",          "CMB-SPA_GREA.yaml"),
    ("CMB-SPA+PP+DESI.yaml",  "grea_Run3_CMB-SPA+PP+DESI",  "CMB-SPA+PP+DESI_GREA.yaml"),
    ("PPS+DESI.yaml",         "grea_Run4_PPS+DESI",         "PPS+DESI_GREA.yaml"),
    ("CMB-SPA+PPS+DESI.yaml", "grea_Run5_CMB-SPA+PPS+DESI", "CMB-SPA+PPS+DESI_GREA.yaml"),
]

GREA_EXTRA = {
    "fluid_equation_of_state": "GREA",
    "cs2_fld": 1.0,
    "use_ppf": "yes",
    "sBBN file": "external/bbn/sBBN_2017.dat",  # prefixed: resolves via the site-packages symlink
}
SQRT_K_ETA0 = {
    "prior":    {"min": 2.5, "max": 4.5},
    "ref":      {"dist": "norm", "loc": 3.48, "scale": 0.1},
    "proposal": 0.05,
    "latex":    r"\sqrt{k}\,\eta_0",
}

for src, folder, dst in RUNS:
    info = yaml_load_file(os.path.join(SRC_DIR, src))
    info["theory"]["classy"].setdefault("extra_args", {})
    info["theory"]["classy"]["extra_args"].update(GREA_EXTRA)   # 1) GREA theory
    info["params"]["sqrt_k_eta0"] = SQRT_K_ETA0                  # 2) sampled GREA param
    info["params"].pop("Omega_Lambda", None)                    #    no Lambda under GREA
    info["output"] = "chains/chains"                            # 3) CCG chain layout
    out_dir = os.path.join(PROJ, folder)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, dst), "w") as f:
        f.write(yaml_dump(info))
    print("wrote", os.path.join(folder, dst))
