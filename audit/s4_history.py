"""s4_history.py - search the repo history for the earlier leak fix.

Runs git log -S for 'leak', 'timing', 'shift' and 'lag' (as asked), the matching diff lines, a working-tree grep, and
pickaxe searches for the tokens that carry the v1 -> v2 change (revlog.factor, cards.factor, shift(, days_since ...).
Writes results/04_history.md and results/04_history.json. Read-only: it never touches the repository.

Usage (repo root):  python audit/s4_history.py
"""
import json
import re
import subprocess

import pandas as pd

import audit_lib as L

L.utf8_stdout()
REPO = L.AUDIT_DIR.parent
OUT, MD = {}, []


def git(*args):
    r = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return r.stdout.strip()


def section(title, body):
    MD.append(f"### {title}\n\n```\n{body}\n```\n")
    print(f"\n## {title}\n{body}")


OUT["head"] = git("rev-parse", "HEAD")
OUT["status_short"] = git("status", "--short")
OUT["dashboard_diff_vs_head"] = git("diff", "HEAD", "--stat", "--", "dashboard.py", "export_data.py")
section("Repository state", f"HEAD {OUT['head']}\ngit status --short: {OUT['status_short'] or '(clean)'}\n"
                            f"git diff HEAD --stat -- dashboard.py export_data.py: {OUT['dashboard_diff_vs_head'] or '(no differences: audited code equals committed code)'}")

full_log = git("log", "--all", "--format=%h %ad %s", "--date=short")
section("Commit log (every commit message in this repo)", full_log)
OUT["n_commits"] = len(full_log.splitlines())
OUT["distinct_messages"] = sorted(set(line.split(" ", 2)[2] for line in full_log.splitlines()))

# 1. the four searches that were asked for
res = {}
lines = []
for term in ["leak", "timing", "shift", "lag"]:
    exact = git("log", "--all", f"-S{term}", "--format=%h %ad %s", "--date=short")
    ci = git("log", "--all", "-i", f"-G{term}", "--format=%h %ad %s", "--date=short")
    res[term] = {"git_log_S": exact.splitlines(), "git_log_i_G": ci.splitlines()}
    lines.append(f"git log --all -S'{term}'      -> {len(exact.splitlines())} commits\n" + ("\n".join("    " + x for x in exact.splitlines()) or "    (none)"))
    lines.append(f"git log --all -i -G'{term}'   -> {len(ci.splitlines())} commits\n" + ("\n".join("    " + x for x in ci.splitlines()) or "    (none)"))
OUT["pickaxe"] = res
section("git log -S / -G for the four terms", "\n".join(lines))

# 2. what the shift/lag hits actually contain (added or removed lines only; notebooks listed separately below)
lines = []
for term in ["shift", "lag"]:
    for c in git("log", "--all", f"-S{term}", "--format=%h").splitlines():
        patch = git("show", c, "--format=", "-p", "--", ".", ":(exclude)*.ipynb", ":(exclude)*.csv", ":(exclude)*.txt", ":(exclude)*.png")
        hit = [ln[:170] for ln in patch.splitlines() if re.match(r"^[+-]", ln) and not ln.startswith(("+++", "---")) and term in ln]
        lines.append(f"term={term} commit={c}: {len(hit)} matching +/- lines (non-notebook files)")
        lines += ["    " + h for h in hit[:6]]
OUT["shift_lag_hits_nonnotebook"] = lines
section("Matching lines inside the shift/lag hit commits", "\n".join(lines))

# 3. working tree grep: code in .py, markdown, and notebook source cells (executed notebooks excluded: they duplicate source + outputs)
grep_hits = []
for path in sorted(REPO.glob("*.py")) + sorted(REPO.glob("*.md")):
    for i, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        if re.search(r"leak|timing|shift|\blag", line, re.I):
            grep_hits.append({"file": path.name, "where": f"line {i}", "text": line.strip()[:150]})
for nb in ["anki_causal_analysis", "psm_response_time", "psm_response_time_v2"]:
    d = json.load(open(REPO / f"{nb}.ipynb", encoding="utf-8"))
    for ci, c in enumerate(d["cells"]):
        for ln, line in enumerate("".join(c["source"]).splitlines(), 1):
            if re.search(r"leak|timing|shift|\blag", line, re.I):
                grep_hits.append({"file": nb + ".ipynb", "where": f"cell {ci} ({c['cell_type']}), line {ln}", "text": line.strip()[:150]})
gdf = pd.DataFrame(grep_hits)
OUT["worktree_grep"] = grep_hits
MD.append("### Working-tree grep for leak|timing|shift|lag (case-insensitive)\n\n" + L.md_table(gdf) + "\n")
print("\n## working-tree grep\n" + L.md_table(gdf))
OUT["worktree_grep_counts"] = {t: int(sum(bool(re.search(t, h["text"], re.I)) for h in grep_hits)) for t in ["leak", "timing", "shift", r"\blag"]}

# 4. the tokens that carry the v1 -> v2 change
tok_lines = []
tok = {}
for t in ["revlog.factor", "cards.factor", "at review time", "current snapshot", "point-in-time", "days_since", "lastIvl", ".shift(", "run_dml"]:
    commits = git("log", "--all", f"-S{t}", "--format=%h %ad", "--date=short").splitlines()
    tok[t] = commits
    tok_lines.append(f"-S'{t}': " + (", ".join(commits) if commits else "(none)"))
OUT["token_pickaxe"] = tok
section("Pickaxe on the tokens that carry the v1 -> v2 change", "\n".join(tok_lines))

# 5. when did v1 and v2 enter git, and how often were they touched afterwards
rows = []
for f in ["psm_response_time.ipynb", "psm_response_time_v2.ipynb", "anki_causal_analysis.ipynb", "psm_response_time_executed.ipynb", "psm_response_time_v2_executed.ipynb"]:
    added = git("log", "--diff-filter=A", "--format=%h %ad", "--date=iso", "--", f)
    touched = git("log", "--format=%h", "--", f).splitlines()
    mtime = pd.Timestamp((REPO / f).stat().st_mtime, unit="s", tz="UTC").tz_convert(L.LOCAL_TZ)
    rows.append({"file": f, "added in commit": added, "commits touching it": len(touched), "file mtime (Eastern)": str(mtime)[:16]})
hdf = pd.DataFrame(rows)
OUT["notebook_history"] = rows
MD.append("### v1 / v2 notebooks: git history against file timestamps\n\n" + L.md_table(hdf) + "\n")
print("\n## notebook history\n" + L.md_table(hdf))

with open(L.RESULTS / "04_history.md", "w", encoding="utf-8") as fh:
    fh.write("\n".join(MD))
L.write_json("04_history.json", OUT)
