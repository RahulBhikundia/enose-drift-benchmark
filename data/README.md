# Data provenance

Downloaded 2026-11-11. Raw files are never modified; all processing writes to `data/processed/`.

## UCI Gas Sensor Array Drift Dataset (`raw/uci_gas_drift/`)
- Source: https://archive.ics.uci.edu/static/public/224/gas+sensor+array+drift+dataset.zip
- DOI: 10.24432/C5RP6W (Vergara et al., 2012)
- Zip SHA-256: `91e8f466f202e7a093d657673ce47311c3e90416f7df3057966058961c351fe4`
- Contents: `Dataset/batch1.dat` … `batch10.dat`, libsvm format (`label idx:value ...`), 128 features, 6 gas classes
- Row counts: b1 445, b2 1244, b3 1586, b4 161, b5 197, b6 2300, b7 3613, b8 294, b9 470, b10 3600 (13,910 in total)
- Labels: 1 Ethanol, 2 Ethylene, 3 Ammonia, 4 Acetaldehyde, 5 Acetone, 6 Toluene
- This release has no concentration column (the related "at different concentrations" dataset, UCI #270, has one).

## SmellNet base data (`raw/smellnet/base_data/`)
- Source: Hugging Face `DeweiFeng/SmellNet`, pinned to commit `45405fec862fcd98cec03215b5e0d1477a377bec`
- License: MIT
- Paper: Feng et al., "SmellNet: A Large-scale Dataset for Real-world Smell Recognition", ICLR 2026, arXiv:2506.00239
- Layout: `training/<substance>/<substance>_{1..5}.csv` and `testing/<substance>/<substance>_6.csv`, giving 50 substances × 6 sessions = 300 files
- Columns: `NO2, C2H5OH, VOC, CO, Alcohol, LPG` (6 channels, integer readings, about 1 Hz, about 570 to 700 rows per session)
- No timestamp column. **Assumption to verify:** the session index 1–6 corresponds to separate collection days in chronological order. The paper says each substance had six sessions on different days, but the files don't confirm the order.
- File list: `raw/smellnet/filelist.json`

## UCI Twin Gas Sensor Arrays (`raw/twin_gas/`)
- Source: https://archive.ics.uci.edu/static/public/361/twin+gas+sensor+arrays.zip (UCI #361; Fonollosa et al., 2016)
- Zip SHA-256: `e8801de3a98d3db2b48fccf0be4d681879a1441f5732ac6fbee1c48a05b49167`
- Contents: `data1/B{unit}_G{gas}_F{conc}_R{rep}.txt`, 640 files; 5 units, 4 gases, 10 concentrations, 100 Hz, 8 MOX sensors
- Features (64 per measurement) are computed by `src/data_twin.py` and cached in `processed/twin_features.csv`.

## Getting the data
`python scripts/data/download_data.py` downloads all three datasets into `data/raw/` and checks the checksums above.
The data are not stored in the repository (about 3 GB unpacked); all three are hosted publicly.
