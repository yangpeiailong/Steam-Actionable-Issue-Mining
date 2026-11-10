from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
credential_pattern = re.compile(r"sk-" + r"[A-Za-z0-9]{20,}")
sensitive_extensions = {".csv", ".xlsx", ".xls", ".jsonl", ".parquet", ".feather"}
errors = []
for path in ROOT.rglob("*"):
    if not path.is_file() or path == Path(__file__).resolve():
        continue
    rel = path.relative_to(ROOT)
    if any(part in {"data_local", "models_local", ".git", "__pycache__"} for part in rel.parts):
        errors.append(f"Local/private directory should not be in a release archive: {rel}")
        continue
    if path.suffix.lower() in sensitive_extensions:
        errors.append(f"Review-level data/output file should not be committed: {rel}")
        continue
    if path.suffix.lower() not in {".py", ".md", ".txt", ".yml", ".yaml", ".toml", ".ini"}:
        continue
    text = path.read_text(encoding="utf-8", errors="ignore")
    if "D:\\BaiduSyncdisk" in text or "D:\\AI_Models" in text:
        errors.append(f"Author-specific absolute path found: {rel}")
    if credential_pattern.search(text):
        errors.append(f"Possible API credential found: {rel}")
if errors:
    print("PUBLIC RELEASE AUDIT FAILED")
    print("\n".join(errors))
    sys.exit(1)
print("PUBLIC RELEASE AUDIT PASSED")
