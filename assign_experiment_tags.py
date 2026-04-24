"""
assign_experiment_tags.py
─────────────────────────
Randomly assigns exp::treatment / exp::control tags to notes in the
"Human Japanese Intermediate Shared" Anki deck for a 60-day self-experiment.

SAFE: opens the database read-only; writes no changes to Anki files.
OUTPUT:
  experiment_tags_import.txt  — Anki-ready tab-separated import file
  experiment_assignment_log.csv — note ID → assignment record for later analysis

Run with Anki CLOSED to avoid database lock errors.
"""

import csv
import random
import sqlite3
import sys
from pathlib import Path

# Force UTF-8 output on Windows so Japanese text and symbols print cleanly
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Configuration ──────────────────────────────────────────────────────────
DB_PATH        = Path(r"C:\Users\sid99\AppData\Roaming\Anki2\Pottapatri\collection.anki2")
TARGET_DECK    = "Human Japanese Intermediate Shared"
TREATMENT_TAG  = "exp::treatment"
CONTROL_TAG    = "exp::control"
RANDOM_SEED    = 42
OUTPUT_DIR     = Path(__file__).parent
IMPORT_FILE    = OUTPUT_DIR / "experiment_tags_import.txt"
LOG_FILE       = OUTPUT_DIR / "experiment_assignment_log.csv"

# ── Helpers ────────────────────────────────────────────────────────────────

def connect_readonly(path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    con.create_collation("unicase", lambda a, b: (a > b) - (a < b))
    con.row_factory = sqlite3.Row
    return con


def abort(msg: str) -> None:
    print(f"\n❌  {msg}", file=sys.stderr)
    sys.exit(1)


# ── Step 1: Connect and confirm deck ───────────────────────────────────────
print("=" * 60)
print("Anki Experiment Tag Assigner")
print("=" * 60)

if not DB_PATH.exists():
    abort(f"Database not found: {DB_PATH}")

try:
    con = connect_readonly(DB_PATH)
except sqlite3.OperationalError as e:
    abort(f"Cannot open database (is Anki running?): {e}")

all_decks = {row["name"]: row["id"] for row in con.execute("SELECT id, name FROM decks")}
print("\nDecks found in database:")
for name in sorted(all_decks):
    print(f"  • {name}")

if TARGET_DECK not in all_decks:
    abort(
        f'Deck "{TARGET_DECK}" not found.\n'
        f'Available decks: {list(all_decks.keys())}'
    )

deck_id = all_decks[TARGET_DECK]
print(f'\n✓  Deck confirmed: "{TARGET_DECK}" (id={deck_id})')

# ── Step 2: Load notes in deck ─────────────────────────────────────────────
# One note can have multiple cards; GROUP BY n.id deduplicates.
rows = con.execute("""
    SELECT
        n.id        AS note_id,
        n.guid      AS guid,
        n.mid       AS mid,
        n.tags      AS tags,
        n.flds      AS flds
    FROM notes n
    JOIN cards c ON c.nid = n.id
    WHERE c.did = ?
    GROUP BY n.id
    ORDER BY n.id
""", (deck_id,)).fetchall()

if not rows:
    abort("No notes found in the target deck.")

print(f"✓  Notes loaded: {len(rows):,}")

# ── Step 3: Safety checks ──────────────────────────────────────────────────
print("\n── Safety checks ──")

# Check for existing exp:: tags
already_tagged = [r for r in rows if "exp::" in (r["tags"] or "")]
if already_tagged:
    abort(
        f"{len(already_tagged)} note(s) already carry an exp:: tag.\n"
        f"Remove them first to avoid double-tagging.\n"
        f"Example note IDs: {[r['note_id'] for r in already_tagged[:5]]}"
    )
print("✓  No existing exp:: tags found")

# Confirm backup intent
print("✓  Existing tags will be preserved (new tag appended, not replaced)")
print(f"✓  Random seed: {RANDOM_SEED}")

# ── Step 4: Collect note-type field definitions ────────────────────────────
# Maps mid → ordered list of field names
mids_used = list({r["mid"] for r in rows})
field_map: dict[int, list[str]] = {}
for mid in mids_used:
    fields = con.execute(
        "SELECT name FROM fields WHERE ntid = ? ORDER BY ord", (mid,)
    ).fetchall()
    field_map[mid] = [f["name"] for f in fields]

notetype_names: dict[int, str] = {
    row["id"]: row["name"]
    for row in con.execute(
        f"SELECT id, name FROM notetypes WHERE id IN ({','.join('?' for _ in mids_used)})",
        mids_used,
    )
}
print(f"✓  Note types in deck: {list(notetype_names.values())}")

con.close()

# ── Step 5: Random assignment ──────────────────────────────────────────────
print("\n── Assigning tags ──")
random.seed(RANDOM_SEED)

note_list = list(rows)
random.shuffle(note_list)
midpoint  = len(note_list) // 2
treatment = set(r["note_id"] for r in note_list[:midpoint])
control   = set(r["note_id"] for r in note_list[midpoint:])

print(f"  Total notes  : {len(note_list):,}")
print(f"  Treatment    : {len(treatment):,}  (exp::treatment)")
print(f"  Control      : {len(control):,}  (exp::control)")

# ── Step 6: Write Anki import file ─────────────────────────────────────────
# Format understood by Anki 2.1.54+ with GUID-based note matching:
#
#   #separator:tab
#   #html:true
#   #guid column:1
#   #notetype column:2
#   #deck column:3
#   #tags column:4
#   guid  notetype  deck  tags  field1  field2  ...
#
# Anki matches on GUID, updates tags and fields in-place, and leaves
# scheduling data (interval, ease, due) completely untouched.

print(f"\n── Writing import file: {IMPORT_FILE.name} ──")

# Anki uses \x1f as the internal field separator inside the flds blob.
ANKI_FIELD_SEP = "\x1f"

with IMPORT_FILE.open("w", encoding="utf-8", newline="\n") as f:
    f.write("#separator:tab\n")
    f.write("#html:true\n")
    f.write("#guid column:1\n")
    f.write("#notetype column:2\n")
    f.write("#deck column:3\n")
    f.write("#tags column:4\n")
    # Comment row describing column layout (Anki ignores lines starting with #)
    f.write("# guid\tnotetype\tdeck\ttags\t[fields...]\n")

    written = 0
    for row in rows:
        note_id   = row["note_id"]
        guid      = row["guid"]
        mid       = row["mid"]
        old_tags  = (row["tags"] or "").strip()
        flds_blob = row["flds"] or ""

        # Assign new tag
        new_tag   = TREATMENT_TAG if note_id in treatment else CONTROL_TAG
        # Append new tag; preserve existing tags, strip duplicate whitespace
        tag_parts = old_tags.split() if old_tags else []
        tag_parts.append(new_tag)
        new_tags  = " ".join(tag_parts)

        # Split fields by Anki's internal separator
        fields    = flds_blob.split(ANKI_FIELD_SEP)

        # Build row: guid | notetype | deck | tags | field1 | field2 | ...
        col_notetype = notetype_names.get(mid, "")
        line_parts   = [guid, col_notetype, TARGET_DECK, new_tags] + fields
        f.write("\t".join(line_parts) + "\n")
        written += 1

print(f"✓  {written:,} rows written to {IMPORT_FILE}")

# ── Step 7: Write assignment log ───────────────────────────────────────────
print(f"── Writing log: {LOG_FILE.name} ──")

with LOG_FILE.open("w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "note_id", "guid", "notetype", "assignment",
        "original_tags", "new_tags"
    ])
    for row in rows:
        note_id  = row["note_id"]
        old_tags = (row["tags"] or "").strip()
        new_tag  = TREATMENT_TAG if note_id in treatment else CONTROL_TAG
        tag_parts = old_tags.split() if old_tags else []
        tag_parts.append(new_tag)
        new_tags = " ".join(tag_parts)
        writer.writerow([
            note_id,
            row["guid"],
            notetype_names.get(row["mid"], ""),
            new_tag,
            old_tags,
            new_tags,
        ])

print(f"✓  {len(rows):,} rows written to {LOG_FILE}")

# ── Done ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("DONE — no changes made to Anki database")
print("=" * 60)
print(f"""
Next step — import into Anki:
  1. Close this script (already done).
  2. Open Anki desktop.
  3. File → Import  (or Ctrl+Shift+I).
  4. Select:  {IMPORT_FILE}
  5. In the import dialog:
       • Import mode   →  "Update existing notes"
       • Match scope   →  "Notetype and deck"  (or "Notetype")
       • Tag modified  →  your choice (safe either way)
  6. Click Import.  Anki will confirm how many notes were updated.
  7. Verify in the Browser (Ctrl+F):  search  exp::treatment  and  exp::control
     to confirm counts match the numbers above.

Experiment log saved to:
  {LOG_FILE}
Keep this file — you will need the note IDs for the analysis after 60 days.
""")
