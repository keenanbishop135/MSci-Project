# Teaching pack

Real EXFOR data, cut from the full X4Pro database so that the notebooks need no large download.
Rebuild it with `python scripts/build_teaching_pack.py --db data/x4sqlite1.db --parquet data/exfor_local_mad_3.parquet`.

| File | Size | What |
|---|---|---|
| `x4_examples.db` | 0.6 MB | A mini X4Pro SQLite file with the **same tables** (`x4pro_ds`, `x4pro_x5z`, `x4pro_hdr`, `x4pro_autocorr`, `REACODE`, `SUBENT`, `ENTRY`, `AUTHORS`, `REFERS`, `INSTITUT`, `DICT034`) restricted to 45 hand-picked datasets. `open_x4`, `MetadataFilter` and the real ingestion code run on it unchanged. |
| `examples.json` | 5 kB | For each dataset: the role it plays and the lesson it teaches (e.g. `raw_counts_labelled_barns`, `dropped_natural_target`, `superseded_in_parquet`). |
| `exfor_teaching.parquet` | 12.5 MB | The rows of the full ingested parquet (with outlier columns and the audit's `barn_class`) for 14 reactions: U-238 (n,f)/(n,g), U-235/Pu-239/Fe-56/Au-197/Lu-176 capture, Fe-56 (n,inl), Cl-35 (n,p), Mo-100 (n,2n), and proton-induced data on Mo-100, Zn-68, Zn-70, Ra-226; for notebook D6 also the total cross sections of Be-9, C-12 and Fe-56 and the (n,g) groups of I-127 and I-129. |
| `capture_cleaned.csv`, `endf_Au197_capture.csv`, `isotope_features.csv` | 0.27 MB | For notebook T1 (the last file: seven nuclear properties per isotope from the AME2020 mass table, for section 11): 7,624 cleaned fast-neutron capture (n,g) points, 0.1 to 20 MeV, of 83 isotopes (entry, Z, A, energy in eV, cross section in barns; at most 300 points per isotope) and the ENDF/B-VII.1 (n,g) curve of Au-197. Rebuilt with `python scripts/build_au197_notebook_data.py`. |
| `np_cleaned.csv`, `endf_Ti47_np.csv`, `endf_U238_capture.csv` | 0.3 MB | For notebook T2: 7,496 cleaned (n,p) points of 80 isotopes (same columns and cleaning as the capture file) and the ENDF/B-VII.1 curves of Ti-47 (n,p) and U-238 (n,g). Rebuilt with `python scripts/build_t2_notebook_data.py`. |

Verified: running `exforlint.ingest.ingest_x4` on `x4_examples.db` reproduces, row for row, the corresponding rows of the full parquet
(2,760 points from 33 datasets; the other 12 datasets are dropped for the documented reasons).
EXFOR is a public compilation (IAEA Nuclear Data Section and partner centres); keep the dataset IDs so every row can be traced to its source.
