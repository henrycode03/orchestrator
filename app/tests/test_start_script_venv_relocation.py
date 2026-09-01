import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).parents[2]


def test_start_repairs_relocated_python_entrypoint_shebang(tmp_path):
    venv_dir = tmp_path / "venv"
    bin_dir = venv_dir / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "python3").symlink_to(sys.executable)

    old_python = tmp_path / "old-location" / "venv" / "bin" / "python3"
    entrypoint = bin_dir / "uvicorn"
    entrypoint.write_text(f"#!{old_python}\n" "import sys\n" "print(sys.executable)\n")
    entrypoint.chmod(0o755)
    (bin_dir / "activate").write_text(
        f"export VIRTUAL_ENV=$(cygpath {old_python.parent.parent})\n"
        f"export VIRTUAL_ENV={old_python.parent.parent}\n"
        f"export VIRTUAL_ENV{old_python.parent.parent}\n"
    )

    repair = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; VENV_DIR="$2"; repair_relocated_venv',
            "test-start-script",
            str(ROOT / "start.sh"),
            str(venv_dir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert repair.returncode == 0, repair.stderr
    assert entrypoint.read_text().splitlines()[0] == f"#!{bin_dir / 'python3'}"
    activation = (bin_dir / "activate").read_text()
    assert f"export VIRTUAL_ENV={venv_dir}" in activation
    assert f"export VIRTUAL_ENV/" not in activation
    assert str(old_python.parent.parent) not in activation

    activated = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; printf "%s" "$VIRTUAL_ENV"',
            "test-activation",
            str(bin_dir / "activate"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert activated.returncode == 0, activated.stderr
    assert activated.stdout == str(venv_dir)

    executed = subprocess.run(
        [str(entrypoint)], capture_output=True, text=True, check=False
    )
    assert executed.returncode == 0, executed.stderr
