"""Generate the plain-script gallery data so the page also works via file://."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
source = root / "docs/assets/images/examples.json"
target = root / "docs/assets/examples.js"
data = json.loads(source.read_text(encoding="utf-8"))
target.write_text(
    "// Generated from images/examples.json by tools/sync_site_data.py.\n"
    "window.DECFLOWEDIT_EXAMPLES = "
    + json.dumps(data, ensure_ascii=False, indent=2)
    + ";\n",
    encoding="utf-8",
)
print("Updated docs/assets/examples.js")
