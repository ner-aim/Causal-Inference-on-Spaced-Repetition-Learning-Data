"""
export_data.py
──────────────
Exports Anki SQLite data to parquet files in the data/ folder.
Run this locally (with Anki closed) before pushing to GitHub,
so the Streamlit Cloud deployment has fresh data to read.

Usage:  python export_data.py
"""

import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

DB_PATH    = Path(r"C:\Users\sid99\AppData\Roaming\Anki2\Pottapatri\collection.anki2")
OUTPUT_DIR = Path(__file__).parent / "data"
OUTPUT_DIR.mkdir(exist_ok=True)

ANKI_CAP_MS = 60_000

print("Connecting to Anki database (must be closed)...")
con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
con.create_collation("unicase", lambda a, b: (a > b) - (a < b))

# ── revlog ──────────────────────────────────────────────────────────────────
print("Exporting revlog...", end=" ")
revlog = pd.read_sql(
    "SELECT id, cid, ease, ivl, factor, time, type FROM revlog", con
)
revlog["ts"]          = pd.to_datetime(revlog["id"], unit="ms")
revlog["date"]        = revlog["ts"].dt.date.astype(str)
revlog["hour"]        = revlog["ts"].dt.hour
revlog["dow"]         = revlog["ts"].dt.dayofweek
revlog["retained"]    = (revlog["ease"] > 1).astype(int)
revlog["ease_factor"] = revlog["factor"] / 1000
revlog["time_s"]      = revlog["time"] / 1000
revlog["capped"]      = (revlog["time"] == ANKI_CAP_MS).astype(int)
revlog.to_parquet(OUTPUT_DIR / "revlog.parquet", index=False)
print(f"{len(revlog):,} rows")

# ── experiment groups (tagged notes → cards → revlog) ──────────────────────
print("Exporting experiment data...", end=" ")
notes = pd.read_sql(
    "SELECT id, tags FROM notes WHERE tags LIKE '%exp::%'", con
)
if len(notes) > 0:
    notes["group"] = notes["tags"].apply(
        lambda t: "treatment" if "exp::treatment" in t else "control"
    )
    note_ids      = notes["id"].tolist()
    placeholders  = ",".join("?" * len(note_ids))
    cards         = pd.read_sql(
        f"SELECT id AS cid, nid FROM cards WHERE nid IN ({placeholders})",
        con, params=note_ids
    )
    note_group = notes.set_index("id")["group"]
    cards["group"] = cards["nid"].map(note_group)

    from datetime import date
    exp_start_ms = int(datetime(2026, 4, 24).timestamp() * 1000)
    cid_list      = cards["cid"].tolist()
    placeholders2 = ",".join("?" * len(cid_list))
    exp_revlog    = pd.read_sql(
        f"SELECT id, cid, ease, ivl, factor, time, type FROM revlog "
        f"WHERE cid IN ({placeholders2}) AND id >= ?",
        con, params=cid_list + [exp_start_ms]
    )
    if len(exp_revlog) > 0:
        exp_revlog["ts"]          = pd.to_datetime(exp_revlog["id"], unit="ms")
        exp_revlog["date"]        = exp_revlog["ts"].dt.date.astype(str)
        exp_revlog["retained"]    = (exp_revlog["ease"] > 1).astype(int)
        exp_revlog["ease_factor"] = exp_revlog["factor"] / 1000
        exp_revlog["capped"]      = (exp_revlog["time"] == ANKI_CAP_MS).astype(int)
        exp_revlog = exp_revlog.merge(cards[["cid", "group"]], on="cid", how="left")
    exp_revlog.to_parquet(OUTPUT_DIR / "experiment_revlog.parquet", index=False)
    print(f"{len(exp_revlog):,} experiment rows")
else:
    pd.DataFrame().to_parquet(OUTPUT_DIR / "experiment_revlog.parquet", index=False)
    print("no tagged notes found")

con.close()

# ── Metadata ────────────────────────────────────────────────────────────────
meta = pd.DataFrame([{"exported_at": datetime.now().isoformat()}])
meta.to_parquet(OUTPUT_DIR / "meta.parquet", index=False)

print(f"\nAll files written to {OUTPUT_DIR}/")
print("  revlog.parquet")
print("  experiment_revlog.parquet")
print("  meta.parquet")
print("\nNext: git add data/ && git push")
