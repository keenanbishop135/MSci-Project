#!/usr/bin/env python
"""Data for the feature-importance and curation notebook (notebooks/T3_Feature_Importance_and_Curation.ipynb).

  python scripts/build_t3_notebook_data.py [--parquet data/exfor_consensus_pure.parquet] [--dates data/v3_selection_cache.parquet]

Writes, in data/teaching/:
  np_linted.csv             every neutron (n,p) point of the linted EXFOR table between 0.1 and 20 MeV, NOT cleaned: one row per measured point with
                            the linter's columns, so that the students do the curation themselves.
                            Columns: dataset (EXFOR subentry, with pointer), entry (first five characters), year_compiled (year in which the entry was compiled into EXFOR, not the year of the measurement;
                            empty if unknown), Z, A, energy_eV, sigma_b, unc_b (the uncertainty as compiled; may be empty),
                            sf8 (reaction modifier, e.g. RAW; empty for a plain cross section), ds_flag (verdict of the consensus on the dataset: ok,
                            normalisation, shape, normalisation+shape, unassessable), ds_offset_dex (offset of the dataset from its peers, log10),
                            ds_weight (weight suggested by the consensus, 0 to 1), pt_outlier (point flagged as a spike).
  nuclide_features.csv      properties of every nuclide of the AME2020 mass table (data/mass_1.mas20.txt) for which they can be computed: the
                            isotope-level features of NucML's published feature sets B1 to B3 that follow from the masses, and two reaction Q-values.
                            Columns: Z, A, N, radius_fm, mass_excess, binding_energy, B_per_A, S_n, S_p, S_2n, S_2p, Q_np, Q_na (MeV, except the
                            radius), N_valence, Z_valence, P_factor, Z_odd, N_odd.
  libraries_Cl35_np.csv     the evaluated Cl-35 (n,p) cross section of ENDF/B-VIII.0 and JENDL-5 at 200 energies from 1.2 to 20 MeV (energy_eV,
                            endfb80_b, jendl5_b), from docs/paper/data/cl35_library_curves.json (PENDF of NJOY2016). Below 1.2 MeV the libraries are a
                            curve of narrow resolved resonances that a table of this kind cannot represent, so the file starts there.

No row is removed for quality: only the reaction (projectile n, MT 103), the energy window, positive values and the absence of a product-state qualifier
select the rows.  The compilation year is taken from the entry date of the linter's selection cache when that file is present."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
EMIN, EMAX = 1e5, 2e7
MAGIC = np.array([2, 8, 20, 28, 50, 82, 126])


def ame_table(path):
    """Mass excess (keV) of every nuclide of the AME2020 mass table, by (Z, A)."""
    rows = []
    for line in open(path, encoding="latin-1").read().splitlines()[36:]:
        try:
            rows.append((int(line[9:14]), int(line[14:19]), float(line[28:42].replace("#", "."))))
        except ValueError:
            continue
    return pd.DataFrame(rows, columns=["Z", "A", "mass_excess"]).set_index(["Z", "A"]).mass_excess


def nuclide_features(me):
    """Isotope-level features from the mass excesses (keV in, MeV out).  A separation energy or Q-value is left empty where a partner nuclide is not in the table."""
    g = lambda z, a: me.get((z, a), np.nan)
    n, h, he4 = g(0, 1), g(1, 1), g(2, 4)
    out = pd.DataFrame([(z, a) for z, a in me.index if z >= 1 and a >= 2], columns=["Z", "A"])
    out["N"] = out.A - out.Z
    out["radius_fm"] = 1.25 * out.A ** (1 / 3)
    za = list(zip(out.Z, out.A))
    out["mass_excess"] = [g(z, a) / 1e3 for z, a in za]
    out["binding_energy"] = [(z * h + (a - z) * n - g(z, a)) / 1e3 for z, a in za]
    out["B_per_A"] = out.binding_energy / out.A
    out["S_n"] = [(g(z, a - 1) + n - g(z, a)) / 1e3 for z, a in za]                 # energy to remove one neutron
    out["S_p"] = [(g(z - 1, a - 1) + h - g(z, a)) / 1e3 for z, a in za]             # ... one proton
    out["S_2n"] = [(g(z, a - 2) + 2 * n - g(z, a)) / 1e3 for z, a in za]
    out["S_2p"] = [(g(z - 2, a - 2) + 2 * h - g(z, a)) / 1e3 for z, a in za]
    out["Q_np"] = [(g(z, a) + n - g(z - 1, a) - h) / 1e3 for z, a in za]            # energy released by (n,p): negative means a threshold
    out["Q_na"] = [(g(z, a) + n - g(z - 2, a - 3) - he4) / 1e3 for z, a in za]      # ... by (n,alpha)
    out["N_valence"] = [int(np.abs(x - MAGIC).min()) for x in out.N]                # distance to the nearest magic number
    out["Z_valence"] = [int(np.abs(x - MAGIC).min()) for x in out.Z]
    s = out.N_valence + out.Z_valence
    out["P_factor"] = np.where(s > 0, out.N_valence * out.Z_valence / s.where(s > 0, 1), 0.0)
    out["Z_odd"], out["N_odd"] = out.Z % 2, out.N % 2
    return out.sort_values(["Z", "A"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet", default=str(ROOT / "data" / "exfor_consensus_pure.parquet"))
    ap.add_argument("--dates", default=str(ROOT / "data" / "v3_selection_cache.parquet"))
    ap.add_argument("--ame", default=str(ROOT / "data" / "mass_1.mas20.txt"))
    ap.add_argument("--libraries", default=str(ROOT / "docs" / "paper" / "data" / "cl35_library_curves.json"))
    a = ap.parse_args()
    dest = ROOT / "data" / "teaching"

    cols = ["experiment_id", "Z", "A", "Energy", "CrossSection", "Uncertainty", "sf8", "ProductState", "ds_flag", "ds_offset_dex", "ds_weight", "pt_outlier"]
    d = pq.read_table(a.parquet, columns=cols, filters=[("Projectile", "=", "n"), ("MT", "=", 103)]).to_pandas()
    n0 = len(d)
    d = d[(d.CrossSection > 0) & (d.Energy >= EMIN) & (d.Energy <= EMAX) & d.ProductState.fillna("").eq("")]          # the reaction and the window; nothing is removed for quality
    d["dataset"] = d.experiment_id.astype(str)
    d["entry"] = d.dataset.str[:5]
    d["year_compiled"] = pd.NA
    if Path(a.dates).exists():
        t = pq.read_table(a.dates, columns=["Entry", "entry_date"], filters=[("Projectile", "=", "n"), ("MT", "=", 103)]).to_pandas().drop_duplicates("Entry")
        year = pd.to_numeric(t.entry_date.astype(str).str[:4], errors="coerce")
        d["year_compiled"] = d.dataset.map(dict(zip(t.Entry.astype(str), year))).astype("Int64")
    out = d.rename(columns={"Energy": "energy_eV", "CrossSection": "sigma_b", "Uncertainty": "unc_b"})
    out = out[["dataset", "entry", "year_compiled", "Z", "A", "energy_eV", "sigma_b", "unc_b", "sf8", "ds_flag", "ds_offset_dex", "ds_weight", "pt_outlier"]]
    out = out.sort_values(["Z", "A", "dataset", "energy_eV"])
    out["ds_offset_dex"], out["ds_weight"] = out.ds_offset_dex.round(4), out.ds_weight.round(4)
    out.to_csv(dest / "np_linted.csv", index=False)
    print(f"(n,p) points: {n0} -> {len(out)} between {EMIN / 1e6:.1f} and {EMAX / 1e6:.0f} MeV ({out.groupby(['Z', 'A']).ngroups} isotopes, {out.dataset.nunique()} datasets, "
          f"{out.entry.nunique()} entries; compilation year known for {out.year_compiled.notna().mean():.0%} of the points)")

    f = nuclide_features(ame_table(a.ame))
    f.round(4).to_csv(dest / "nuclide_features.csv", index=False)
    have = out[["Z", "A"]].drop_duplicates().merge(f, on=["Z", "A"], how="left")
    print(f"nuclide features: {len(f)} nuclides, columns {list(f.columns[2:])}; isotopes of the (n,p) table with an empty feature: {int(have.isna().any(axis=1).sum())}")

    lib = json.load(open(a.libraries))
    E = np.logspace(np.log10(1.2e6), np.log10(EMAX), 200)
    curve = {}
    for name, col in (("ENDF-B-VIII.0", "endfb80_b"), ("JENDL-5", "jendl5_b")):
        e, s = np.array(lib[name]["E"], float), np.array(lib[name]["S"], float)
        ok = (e > 0) & (s > 0)
        curve[col] = 10 ** np.interp(np.log10(E), np.log10(e[ok]), np.log10(s[ok]))
    pd.DataFrame({"energy_eV": E, **curve}).to_csv(dest / "libraries_Cl35_np.csv", index=False)
    print("libraries: Cl-35 (n,p) of ENDF/B-VIII.0 and JENDL-5 at 200 energies from 1.2 to 20 MeV")


if __name__ == "__main__":
    main()
