"""Genera kaggle/out/<nombre>.py: cabecera común (clona el repo y fija el commit) + script de control.

    python kaggle/build_driver.py razonador_t4            # usa el commit HEAD
El commit fijado queda registrado en results/entorno.json de cada notebook.
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
name = sys.argv[1]
commit = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "HEAD" else subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip()
header = (HERE / "drivers" / "common_header.py").read_text().replace("@@COMMIT@@", commit)
text = (HERE / "drivers" / f"{name}.py").read_text().replace("#@@HEADER@@\n", header)
for arg in sys.argv[3:]:  # KEY=VALOR sustituye @@KEY@@
    k, v = arg.split("=", 1)
    text = text.replace(f"@@{k}@@", v)
assert "@@" not in text, "quedan marcadores sin sustituir"
out = HERE / "out" / f"{name}.py"
out.parent.mkdir(exist_ok=True)
out.write_text(text)
print(out, len(text), "caracteres, commit", commit[:12])
