"""Download the three public datasets into data/raw/ and verify their checksums.

  python scripts/data/download_data.py

The datasets are not stored in the repository (about 3 GB unpacked). All three are publicly hosted:
  UCI #224  Gas Sensor Array Drift      https://archive.ics.uci.edu/dataset/224   (zip, SHA-256 checked)
  UCI #361  Twin Gas Sensor Arrays      https://archive.ics.uci.edu/dataset/361   (zip, SHA-256 checked)
  SmellNet  Hugging Face DeweiFeng/SmellNet, base_data/ only, pinned to one commit
"""
import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"

UCI = [  # (folder, url, zip name, sha256)
    ("uci_gas_drift", "https://archive.ics.uci.edu/static/public/224/gas+sensor+array+drift+dataset.zip",
     "gas_sensor_array_drift.zip", "91e8f466f202e7a093d657673ce47311c3e90416f7df3057966058961c351fe4"),
    ("twin_gas", "https://archive.ics.uci.edu/static/public/361/twin+gas+sensor+arrays.zip",
     "twin_gas_sensor_arrays.zip", "e8801de3a98d3db2b48fccf0be4d681879a1441f5732ac6fbee1c48a05b49167"),
]
SMELLNET_REPO, SMELLNET_REV = "DeweiFeng/SmellNet", "45405fec862fcd98cec03215b5e0d1477a377bec"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def get_uci():
    for folder, url, name, digest in UCI:
        d = RAW / folder; d.mkdir(parents=True, exist_ok=True); z = d / name
        if not z.exists():
            print("downloading", url); urllib.request.urlretrieve(url, z)
        got = sha256(z)
        if got != digest:
            sys.exit(f"checksum mismatch for {z}: {got}")
        with zipfile.ZipFile(z) as f:
            f.extractall(d)          # -> Dataset/batch*.dat  or  data1/B*_G*_F*_R*.txt
        print("ok", folder)


def get_smellnet():
    from huggingface_hub import snapshot_download
    d = RAW / "smellnet"
    snapshot_download(SMELLNET_REPO, repo_type="dataset", revision=SMELLNET_REV, local_dir=d,
                      allow_patterns=["base_data/*"])
    (d / "commit.txt").write_text(f"commit sha: {SMELLNET_REV}\n")
    n = len(list((d / "base_data").rglob("*.csv")))
    print(f"ok smellnet ({n} csv files; expected 300)")


if __name__ == "__main__":
    get_uci(); get_smellnet()
