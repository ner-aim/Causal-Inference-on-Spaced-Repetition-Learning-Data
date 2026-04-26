"""
mark_treatment_cards.py
───────────────────────
Prepends a visible marker to the English (Meaning) field of all
treatment-group notes so you can identify them during review and
apply the treatment behaviour (reading the sentence out loud).

Run with Anki CLOSED:
    python mark_treatment_cards.py          # add markers
    python mark_treatment_cards.py --undo   # remove markers
"""

import sys
import sqlite3
import shutil
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DB_PATH = Path(r"C:\Users\sid99\AppData\Roaming\Anki2\Pottapatri\collection.anki2")
MARKER        = "★ "     # prepended to marked fields
MARK_FIELDS   = [1, 2]   # 1=WrittenForm (Japanese), 2=Meaning (English)
FIELD_SEP = "\x1f"

UNDO = "--undo" in sys.argv


def backup(db: Path) -> Path:
    stamp  = time.strftime("%Y%m%d_%H%M%S")
    target = db.parent / f"collection_backup_{stamp}.anki2"
    shutil.copy2(db, target)
    return target


def main():
    if not DB_PATH.exists():
        print(f"ERROR: database not found at {DB_PATH}")
        sys.exit(1)

    bak = backup(DB_PATH)
    print(f"Backup created: {bak.name}")

    con = sqlite3.connect(str(DB_PATH))
    con.create_collation("unicase", lambda a, b: (a > b) - (a < b))

    rows = con.execute(
        "SELECT id, flds FROM notes WHERE tags LIKE '%exp::treatment%'"
    ).fetchall()
    print(f"Treatment notes found: {len(rows):,}")

    updated = 0
    skipped = 0
    now     = int(time.time())

    for note_id, flds_raw in rows:
        fields   = flds_raw.split(FIELD_SEP)
        changed  = False

        for idx in MARK_FIELDS:
            if UNDO:
                if fields[idx].startswith(MARKER):
                    fields[idx] = fields[idx][len(MARKER):]
                    changed = True
            else:
                if not fields[idx].startswith(MARKER):
                    fields[idx] = MARKER + fields[idx]
                    changed = True

        if changed:
            updated += 1
            new_flds = FIELD_SEP.join(fields)
            con.execute(
                "UPDATE notes SET flds=?, mod=?, usn=-1 WHERE id=?",
                (new_flds, now, note_id)
            )
        else:
            skipped += 1

    con.commit()
    con.close()

    action = "Removed marker from" if UNDO else "Added marker to"
    print(f"{action} {updated:,} notes  ({skipped:,} already {'without' if UNDO else 'with'} marker — skipped)")
    print("\nDone. Open Anki — it will sync the changes automatically.")


if __name__ == "__main__":
    main()
