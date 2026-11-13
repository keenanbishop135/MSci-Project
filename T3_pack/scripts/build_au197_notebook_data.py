#!/usr/bin/env python
"""Data for the first-decision-tree notebook (notebooks/T1_First_Decision_Tree_Au197.ipynb): fast neutron capture, (n,gamma).

  python scripts/build_au197_notebook_data.py [--parquet data/exfor_consensus_pure.parquet] [--endf-zip data/external/2026-10-08/ENDF-B-VII.1-neutrons.zip]

Writes, in data/teaching/:
  capture_cleaned.csv       neutron capture points of EXFOR, 0.1 to 20 MeV, cleaned: entry, Z, A, energy_eV, sigma_b
  endf_Au197_capture.csv    the ENDF/B-VII.1 evaluated capture cross section of Au-197 at 150 energies, 0.1 to 20 MeV (energy_eV, sigma_b)
  isotope_features.csv      seven properties of each isotope of the capture file, from the AME2020 mass table (data/mass_1.mas20.txt): N, Q_capture, S_n, B_per_A, Z_odd, N_odd, N_magic_dist (the features of section 11 of T1)

Cleaning (each step is one line below): a plain cross section only (no SF5, SF8 or SF9 modifier, no product-state qualifier, positive energy and value);
no dataset that the statistical consensus flags as off in normalisation or shape, and no point flagged as an outlier; the energy window 0.1 to 20 MeV
(below it the cross section is dominated by resonances and the evaluation needs a reconstruction that the file does not contain);
isotopes with at least 30 points left; at most 300 points per isotope (a random sample, seed 0), so that no isotope dominates."""
import argparse
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
EMIN, EMAX, MINPTS, CAP = 1e5, 2e7, 30, 300
EVAL_FROM = EMIN         # the resolved resonance region of ENDF/B-VII.1 Au-197 ends at 5 keV; above it the file is a plain table of cross sections


def _f(s):
    """One 11-character ENDF float field ('1.234567+3' means 1.234567e+3)."""
    s = s.strip()
    if not s:
        return 0.0
    for i in range(len(s) - 1, 0, -1):
        if s[i] in "+-" and s[i - 1] not in "eE":
            return float(s[:i] + "e" + s[i:])
    return float(s)


def endf_mf3(text, mt):
    """Energies (eV), cross sections (b), interpolation boundaries and laws of the MF3 section of reaction MT of an ENDF-6 file."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if len(line) >= 75 and line[70:72].strip() == "3" and line[72:75].strip() == str(mt):
            nr, npts = int(lines[i + 1][44:55]), int(lines[i + 1][55:66])
            nrl = (nr + 2) // 3
            v = []
            for k in range(nrl):
                row = lines[i + 2 + k]
                v += [int(row[11 * m:11 * m + 11]) for m in range(6) if row[11 * m:11 * m + 11].strip()]
            vals = []
            for k in range((npts * 2 + 5) // 6):
                row = lines[i + 2 + nrl + k]
                vals += [_f(row[11 * m:11 * m + 11]) for m in range(6) if row[11 * m:11 * m + 11].strip()]
            a = np.array(vals[:2 * npts]).reshape(-1, 2)
            return a[:, 0], a[:, 1], v[0::2], v[1::2]
    raise KeyError(mt)


def endf_eval(E, S, nbt, law, Eq):
    """Evaluate an ENDF table at Eq with the interpolation law of each interval (1 histogram, 2 linear-linear, 3 linear-log x, 4 log y-linear x, 5 log-log)."""
    out = np.full(len(Eq), np.nan)
    region = np.searchsorted(np.array(nbt), np.arange(len(E)) + 1, side="left")
    for q, e in enumerate(Eq):
        k = int(np.searchsorted(E, e, side="right")) - 1
        if k < 0 or k >= len(E) - 1:
            continue
        lw, (x0, x1), (y0, y1) = law[min(region[k + 1], len(law) - 1)], E[k:k + 2], S[k:k + 2]
        f = (e - x0) / (x1 - x0)
        if lw == 1:
            out[q] = y0
        elif lw == 2 or y0 <= 0 or y1 <= 0:
            out[q] = y0 + f * (y1 - y0)
        elif lw == 3:
            out[q] = y0 + (y1 - y0) * np.log(e / x0) / np.log(x1 / x0)
        elif lw == 4:
            out[q] = np.exp(np.log(y0) + f * (np.log(y1) - np.log(y0)))
        else:
            out[q] = np.exp(np.log(y0) + np.log(e / x0) / np.log(x1 / x0) * (np.log(y1) - np.log(y0)))
    return out


def ame_table(path):
    """Mass excess (keV) and binding energy per nucleon (keV) of every nuclide of the AME2020 mass table, by (Z, A)."""
    rows = []
    for line in open(path, encoding="latin-1").read().splitlines()[36:]:
        try:
            rows.append((int(line[9:14]), int(line[14:19]), float(line[28:42].replace("#", ".")), float(line[54:67].replace("#", "."))))
        except ValueError:
            continue
    return pd.DataFrame(rows, columns=["Z", "A", "mass_excess", "b_per_a"]).set_index(["Z", "A"])


def isotope_features(iso, ame):
    """Properties of the isotopes (a frame with columns Z, A): neutron number, energy released by capturing a neutron (the neutron separation energy of the compound nucleus), neutron
    separation energy of the target, binding energy per nucleon (all in MeV), parities of Z and N, and the distance of N to the nearest magic number."""
    n_me = ame.mass_excess[(0, 1)]                                                       # the neutron
    me = lambda z, a: ame.mass_excess.get((z, a), np.nan)
    out = iso[["Z", "A"]].drop_duplicates().sort_values(["Z", "A"]).reset_index(drop=True)
    out["N"] = out.A - out.Z
    out["Q_capture"] = [(me(z, a) + n_me - me(z, a + 1)) / 1000 for z, a in zip(out.Z, out.A)]
    out["S_n"] = [(me(z, a - 1) + n_me - me(z, a)) / 1000 for z, a in zip(out.Z, out.A)]
    out["B_per_A"] = [ame.b_per_a.get((z, a), np.nan) / 1000 for z, a in zip(out.Z, out.A)]
    out["Z_odd"], out["N_odd"] = out.Z % 2, out.N % 2
    magic = np.array([2, 8, 20, 28, 50, 82, 126])
    out["N_magic_dist"] = [int(np.abs(n - magic).min()) for n in out.N]
    assert out.notna().all().all(), "an isotope is missing from the mass table"
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet", default=str(ROOT / "data" / "exfor_consensus_pure.parquet"))
    ap.add_argument("--ame", default=str(ROOT / "data" / "mass_1.mas20.txt"))
    ap.add_argument("--endf-zip", default=str(ROOT / "data" / "external" / "2026-10-08" / "ENDF-B-VII.1-neutrons.zip"))
    a = ap.parse_args()
    cols = ["Entry", "Z", "A", "Energy", "CrossSection", "sf5", "sf8", "sf9", "ProductState", "ds_flag", "pt_outlier"]
    d = pq.read_table(a.parquet, columns=cols + ["Projectile", "MT"], filters=[("Projectile", "=", "n"), ("MT", "=", 102)]).to_pandas()
    for c in ("sf5", "sf8", "sf9", "ProductState"):
        d[c] = d[c].replace("", None)
    n0 = len(d)
    d = d[d.sf5.isna() & d.sf8.isna() & d.sf9.isna() & d.ProductState.isna() & (d.CrossSection > 0) & (d.Energy > 0)]      # a plain cross section
    n1 = len(d)
    d = d[~d.ds_flag.isin(["normalisation", "shape", "normalisation+shape"]) & ~d.pt_outlier.fillna(False).astype(bool)]     # not an outlier dataset or point
    n2 = len(d)
    d = d[(d.Energy >= EMIN) & (d.Energy <= EMAX)]                                                                          # the fast region
    d = d[d.groupby(["Z", "A"]).Energy.transform("size") >= MINPTS]                                                         # enough points per isotope
    d = d.sample(frac=1, random_state=0).groupby(["Z", "A"]).head(CAP)                                                      # at most CAP random points per isotope, so that none dominates
    out = d[["Entry", "Z", "A", "Energy", "CrossSection"]].rename(columns={"Entry": "entry", "Energy": "energy_eV", "CrossSection": "sigma_b"})
    out = out.sort_values(["Z", "A", "entry", "energy_eV"])
    dest = ROOT / "data" / "teaching"
    out.to_csv(dest / "capture_cleaned.csv", index=False)
    print(f"(n,gamma) points: {n0} -> plain {n1} -> not flagged {n2} -> final {len(out)} ({out.groupby(['Z', 'A']).ngroups} isotopes, {out.entry.nunique()} entries)")
    z = zipfile.ZipFile(a.endf_zip)
    name = next(n for n in z.namelist() if re.match(r"neutrons/n-079_Au_197\.endf", n))
    E, s, nbt, law = endf_mf3(z.read(name).decode("latin-1"), 102)
    Eg = np.logspace(np.log10(EVAL_FROM), np.log10(EMAX), 150)
    pd.DataFrame({"energy_eV": Eg, "sigma_b": endf_eval(E, s, nbt, law, Eg)}).to_csv(dest / "endf_Au197_capture.csv", index=False)
    print(f"ENDF/B-VII.1 Au-197 (n,gamma) evaluated at 150 energies from {EVAL_FROM / 1e6:.1f} to {EMAX / 1e6:.0f} MeV")
    feats = isotope_features(out, ame_table(a.ame))
    feats.round(4).to_csv(dest / "isotope_features.csv", index=False)
    print(f"isotope features: {len(feats)} isotopes, columns {list(feats.columns[2:])}")


if __name__ == "__main__":
    main()
