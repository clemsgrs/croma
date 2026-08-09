"""Distribution-boundary contracts that a repository import cannot prove."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_benchmark_plot_identity_loads_without_paper_tooling(tmp_path: Path) -> None:
    """The packaged benchmark layer must not reach into excluded paper generators."""
    scripts = tmp_path / "scripts"
    shutil.copytree(ROOT / "scripts" / "bench", scripts / "bench")
    shutil.copytree(ROOT / "src", tmp_path / "src")
    program = """
import json
from plotting import style

print(json.dumps({
    "order_tail": style.CANONICAL_MODEL_ORDER[-6:],
    "families": {
        model: style.MODEL_FAMILY_MAP[model]
        for model in ["RudolfV 2", "RudolfV 2-B", "RudolfV 2-S", "Mascaret", "Phaet"]
    },
}))
"""

    completed = subprocess.run(
        [sys.executable, "-c", program],
        check=False,
        capture_output=True,
        cwd=tmp_path,
        text=True,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join([str(scripts / "bench"), str(tmp_path / "src")]),
        },
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {
        "order_tail": [
            "RudolfV 2",
            "RudolfV 2-B",
            "RudolfV 2-S",
            "Mascaret",
            "Phaet",
            "DINOv2-B",
        ],
        "families": {
            "RudolfV 2": "rudolfv2",
            "RudolfV 2-B": "rudolfv2",
            "RudolfV 2-S": "rudolfv2",
            "Mascaret": "waiv",
            "Phaet": "waiv",
        },
    }
