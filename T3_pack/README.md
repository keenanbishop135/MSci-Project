# T3 pack: feature importance (week 5)

This folder holds everything behind the T3 notebook, so that you can see where each number comes from.

## To start
1. Python 3.10 or newer. Install the libraries once: `pip install -r requirements_T3.txt` (the one new library is `shap`).
2. Open `notebooks/T3_Feature_Importance_and_Curation.ipynb` from the `notebooks/` folder (or from this folder) and run it from the top. It finds the data in `data/teaching/`.

## What is in here
| file | what it is | where it comes from |
|---|---|---|
| `notebooks/T3_Feature_Importance_and_Curation.ipynb` | the workbook | |
| `data/teaching/capture_cleaned.csv` | cleaned fast-neutron capture (n,gamma) points, 83 isotopes (as in T1) | EXFOR (X4Pro, November 2025), cleaned by `scripts/build_au197_notebook_data.py` |
| `data/teaching/isotope_features.csv` | seven properties of those isotopes (as in T1) | AME2020, `scripts/build_au197_notebook_data.py` |
| `data/teaching/endf_Au197_capture.csv` | ENDF/B-VII.1 evaluated Au-197 (n,gamma) curve | ENDF/B-VII.1 |
| `data/teaching/nuclide_features.csv` | 16 properties of 3,556 nuclides computed from the AME2020 masses (mass excess, binding energy, separation energies, Q-values, valence nucleons, parity flags, ...) | `data/mass_1.mas20.txt` through `scripts/build_t3_notebook_data.py` |
| `data/mass_1.mas20.txt` | the AME2020 atomic mass table, as published by the Atomic Mass Data Center | Wang et al., Chinese Physics C 45 (2021) 030003 |
| `scripts/build_t3_notebook_data.py`, `scripts/build_au197_notebook_data.py` | the code that wrote the CSV files above | |

## About the scripts
They are included so that nothing is hidden: read them to see exactly which rows were selected and how each property was computed. `build_t3_notebook_data.py` also writes two files that are used next week and are not in this pack. To *rerun* them you would also need the full EXFOR table and the evaluated-library files, which are not in this pack, so you do not need to run them. Everything the notebook needs is in `data/teaching/`.

## Rules
* Every error is a factor (RMS factor and median factor of 10^|log10 prediction - log10 measurement|), as in T1.
* Hold out whole isotopes when you validate. Choose features and settings without looking at your test isotope.
* Say in your report what you did and what you did not check.
