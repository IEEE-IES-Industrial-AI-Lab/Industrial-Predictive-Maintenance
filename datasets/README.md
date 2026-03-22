# Datasets

## How to Get the Data

### NASA CMAPSS (Required for all RUL benchmarks)

**Important:** There are TWO NASA turbofan datasets. Make sure you use the right one:

| Dataset | Year | Use case |
|---------|------|----------|
| **CMAPSS** | 2008 | This repo — FD001–FD004 subsets |
| **N-CMAPSS** | 2021 | More recent papers — see note below |
```bash
# CMAPSS auto-downloads when you run the loader:
python -c "from datasets.cmapss_loader import CMAPSSLoader; CMAPSSLoader().load()"

# Or manual download:
# https://data.nasa.gov/dataset/C-MAPSS-Aircraft-Engine-Simulator-Data
# Extract all .txt files to: data/cmapss/
```

**N-CMAPSS (2021 update)** — used in recent TII papers:
- Source: https://data.nasa.gov/dataset/N-CMAPSS
- Not yet integrated; contributions welcome (see Contributing)

---

### CWRU Bearing Dataset

**Important:** Motor load condition MUST match when comparing to published baselines.
Most papers use **0 HP load (1797 RPM)**. Always specify which load you used.
```bash
python datasets/cwru_loader.py --download --data-dir data/cwru
# Source: https://engineering.case.edu/bearingdatacenter
```

Fault categories: Normal, Inner Race, Outer Race, Ball  
Fault sizes: 0.007", 0.014", 0.021" diameter

---

### IMS Bearing Dataset
```bash
# Manual download required (free, no login):
# https://ti.arc.nasa.gov/c/3/
# Extract to: data/ims/
```

---

### Paderborn Bearing Dataset

**Important:** Use the official Paderborn train/test split protocol for fair comparison.
Mixing artificial and real damage conditions without acknowledging it is a common error.

- Source: https://mb.uni-paderborn.de/kat/forschung/kat-datacenter/bearing-datacenter/
- 32 operating conditions, artificial + real damage
- Manual download required (free, registration needed)

---

### SWaT / WADI

**Important:** These datasets require a formal data request to iTrust, Singapore University.
They **cannot** be downloaded automatically. Do not commit data files to the repo.

- Application form: https://itrust.sutd.edu.sg/itrust-labs_datasets/
- SWaT: 51 sensors, water treatment plant
- WADI: 123 sensors, water distribution system

---

### MVTec AD (for Smart-Manufacturing-AI repo only)

**License warning:** MVTec AD has a **non-commercial research license**.
It cannot be redistributed. Loaders must point to the official download source:
- https://www.mvtec.com/company/research/datasets/mvtec-ad

---

## Data Directory Structure

After downloading, your `data/` directory should look like:
