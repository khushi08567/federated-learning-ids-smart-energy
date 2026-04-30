import pandas as pd
import glob
import os

CICIOT_PATH = "data/raw/CICIoT2023/wataiData/csv/CICIoT2023/"
EDGE_PATH   = "data/raw/EdgeIIoTset/Edge-IIoTset dataset/Selected dataset for ML and DL/DNN-EdgeIIoT-dataset.csv"

# Check CICIoT2023 columns
print("=" * 60)
print("CICIoT2023 COLUMNS")
print("=" * 60)
first_cic = glob.glob(os.path.join(CICIOT_PATH, "*.csv"))[0]
print(f"Reading: {os.path.basename(first_cic)}")
cic = pd.read_csv(first_cic, nrows=5, low_memory=False)
print(f"Total columns: {len(cic.columns)}")
for col in cic.columns:
    print(f"  {col}  →  dtype: {cic[col].dtype}")

print("\n")

# Check EdgeIIoTset columns
print("=" * 60)
print("EdgeIIoTset COLUMNS")
print("=" * 60)
edge = pd.read_csv(EDGE_PATH, nrows=5, low_memory=False)
print(f"Total columns: {len(edge.columns)}")
for col in edge.columns:
    print(f"  {col}  →  dtype: {edge[col].dtype}")