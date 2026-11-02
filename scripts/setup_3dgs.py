#!/usr/bin/env python3
"""
AeroSplat - 3D Gaussian Splatting Setup Script

This script:
1. Checks that Git is installed.
2. Checks the Python version.
3. Clones the official Graphdeco-Inria 3DGS repository.
4. Initializes/updates the repository submodules.
5. Prints the recommended installation commands.
6. Does NOT silently modify the user's global Python environment.

Recommended usage:
    python scripts/setup_3dgs.py

Optional:
    python scripts/setup_3dgs.py --target external/gaussian-splatting
    python scripts/setup_3dgs.py --force
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


# Official repository
REPO_URL = "https://github.com/graphdeco-inria/gaussian-splatting.git"

# Repository directory relative to AeroSplat project root
DEFAULT_TARGET = Path("external") / "gaussian-splatting"


def print_header() -> None:
    print("=" * 70)
    print("AeroSplat - 3D Gaussian Splatting Setup")
    print("=" * 70)
    print()


def command_exists(command: str) -> bool:
    """Return True if a command exists on PATH."""
    return shutil.which(command) is not None


def run_command(
    command: list[str],
    cwd: Path | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """
    Run a subprocess while displaying the command.
    """
    print(f"\n$ {' '.join(command)}")

    return subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        check=check,
        text=True,
    )


def find_project_root() -> Path:
    """
    Determine AeroSplat project root.

    Expected layout:

        AeroSplat/
        ├── scripts/
        │   └── setup_3dgs.py
        ├── main.py
        └── ...
    """
    script_path = Path(__file__).resolve()

    # scripts/setup_3dgs.py -> project root
    project_root = script_path.parent.parent

    return project_root


def check_python_version() -> bool:
    """
    Check Python version.

    The reference 3DGS code has historically been most comfortable
    with Python 3.8-3.11 depending on the CUDA/PyTorch environment.

    We warn rather than hard-fail because existing environments vary.
    """
    major = sys.version_info.major
    minor = sys.version_info.minor

    print(f"Python version detected: {major}.{minor}.{sys.version_info.micro}")

    if major != 3:
        print("ERROR: Python 3 is required.")
        return False

    if minor < 8:
        print("WARNING: Python < 3.8 may not be suitable for the 3DGS dependencies.")

    if minor >= 12:
        print(
            "WARNING: Python 3.12+ may cause compatibility issues with "
            "some versions of the reference 3DGS dependencies."
        )

    return True


def check_git() -> bool:
    """Check whether Git is available."""
    if not command_exists("git"):
        print(
            "ERROR: Git was not found on PATH.\n"
            "Install Git first and run this script again."
        )
        return False

    result = subprocess.run(
        ["git", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )

    print(f"Git: {result.stdout.strip()}")

    return result.returncode == 0


def check_cuda() -> None:
    """
    Check CUDA availability.

    This does not require CUDA to be installed because the user may
    use a CPU-only environment for repository setup.
    """
    if command_exists("nvidia-smi"):
        print("\nNVIDIA GPU detected:")
        subprocess.run(
            ["nvidia-smi"],
            check=False,
        )
    else:
        print(
            "\nWARNING: nvidia-smi was not found.\n"
            "The setup can continue, but CUDA acceleration may not be available."
        )


def clone_repository(target: Path, force: bool = False) -> bool:
    """
    Clone the official Graphdeco-Inria repository.

    If the directory already exists:
      - If it is a Git repository, update it.
      - Otherwise fail unless --force was supplied.
    """
    if target.exists():
        git_dir = target / ".git"

        if git_dir.exists():
            print(f"\n3DGS repository already exists: {target}")

            print("Updating existing repository...")

            try:
                run_command(
                    ["git", "pull", "--ff-only"],
                    cwd=target,
                )

                run_command(
                    ["git", "submodule", "update", "--init", "--recursive"],
                    cwd=target,
                )

                return True

            except subprocess.CalledProcessError as exc:
                print(f"WARNING: Could not update existing repository: {exc}")
                return False

        if not force:
            print(
                f"\nERROR: Target directory already exists and is not a Git repository:\n"
                f"    {target}\n\n"
                "Choose another target with --target or use --force."
            )
            return False

        print(f"\nRemoving existing directory because --force was supplied: {target}")
        shutil.rmtree(target)

    target.parent.mkdir(parents=True, exist_ok=True)

    try:
        run_command(
            [
                "git",
                "clone",
                "--recurse-submodules",
                REPO_URL,
                str(target),
            ]
        )
    except subprocess.CalledProcessError:
        print("\nERROR: Failed to clone Graphdeco-Inria 3DGS.")
        return False

    return True


def detect_pip() -> list[str]:
    """
    Use the current Python interpreter for pip.
    """
    return [sys.executable, "-m", "pip"]


def upgrade_pip() -> bool:
    """
    Upgrade pip inside the active environment.

    This is intentionally explicit and limited to the active
    Python environment.
    """
    try:
        run_command(
            detect_pip() + ["install", "--upgrade", "pip"],
        )
        return True
    except subprocess.CalledProcessError:
        print("WARNING: pip upgrade failed.")
        return False


def install_3dgs_requirements(repo_dir: Path) -> bool:
    """
    Try to install the Python requirements from the cloned repository.

    The official repository may rely on additional CUDA/C++ packages
    which can fail to build depending on the local environment.
    Therefore this function reports failures instead of pretending
    everything installed successfully.
    """
    requirements_file = repo_dir / "requirements.txt"

    if not requirements_file.exists():
        print(
            "\nNo requirements.txt was found in the cloned 3DGS repository."
        )
        return False

    print(f"\nInstalling Python dependencies from: {requirements_file}")

    try:
        run_command(
            detect_pip()
            + [
                "install",
                "-r",
                str(requirements_file),
            ]
        )
        return True

    except subprocess.CalledProcessError:
        print(
            "\nWARNING: Some 3DGS dependencies could not be installed.\n"
            "This is commonly related to CUDA, PyTorch, or native C++/CUDA "
            "extensions."
        )
        return False


def print_manual_build_steps(repo_dir: Path) -> None:
    """
    Print commands for native CUDA extensions that may require
    manual installation.
    """
    print("\n" + "=" * 70)
    print("3DGS repository setup completed")
    print("=" * 70)

    print("\nRepository:")
    print(f"  {repo_dir}")

    print("\nRecommended next steps:")
    print()

    if os.name == "nt":
        print("PowerShell:")
        print(f'  cd "{repo_dir}"')
    else:
        print("Shell:")
        print(f'  cd "{repo_dir}"')

    print(
        "\nIf the repository's native dependencies have not been installed, "
        "follow its README installation section."
    )

    print(
        "\nFor AeroSplat, the expected workflow is:"
        "\n  Drone Images"
        "\n      -> COLMAP"
        "\n      -> cameras + sparse reconstruction"
        "\n      -> Graphdeco-Inria 3DGS"
        "\n      -> trained Gaussian scene"
        "\n      -> AeroSplat analysis"
    )

    print("\nIMPORTANT:")
    print(
        "Do not run the 3DGS training command until COLMAP has produced "
        "a valid sparse reconstruction for the selected scene."
    )

    print("\n" + "=" * 70)


def verify_repository(repo_dir: Path) -> bool:
    """
    Verify important files/directories in the cloned repository.
    """
    if not repo_dir.exists():
        return False

    required_items = [
        repo_dir / ".git",
    ]

    for item in required_items:
        if not item.exists():
            print(f"WARNING: Expected item not found: {item}")
            return False

    print("\nRepository verification:")
    print(f"  [OK] {repo_dir}")

    # These are informational because repository versions can change.
    for relative in [
        "train.py",
        "convert.py",
        "scene",
        "gaussian_renderer",
    ]:
        path = repo_dir / relative
        status = "[OK]" if path.exists() else "[INFO]"
        print(f"  {status} {relative}")

    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Set up Graphdeco-Inria 3D Gaussian Splatting for AeroSplat."
    )

    parser.add_argument(
        "--target",
        type=Path,
        default=DEFAULT_TARGET,
        help=(
            "Directory where the Graphdeco-Inria repository should be cloned. "
            f"Default: {DEFAULT_TARGET}"
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete an existing non-Git target directory before cloning.",
    )

    parser.add_argument(
        "--install",
        action="store_true",
        help="Attempt to install the repository Python requirements.",
    )

    parser.add_argument(
        "--no-update",
        action="store_true",
        help="Do not pull updates when the repository already exists.",
    )

    args = parser.parse_args()

    print_header()

    project_root = find_project_root()

    print(f"Project root: {project_root}")

    if not check_python_version():
        return 1

    if not check_git():
        return 1

    check_cuda()

    target = args.target

    if not target.is_absolute():
        target = project_root / target

    target = target.resolve()

    print(f"\n3DGS target directory: {target}")

    # Existing repository case
    if target.exists() and (target / ".git").exists() and args.no_update:
        print("\nExisting repository detected; --no-update was supplied.")
        verified = verify_repository(target)

        if not verified:
            return 1

    else:
        success = clone_repository(
            target=target,
            force=args.force,
        )

        if not success:
            return 1

        verified = verify_repository(target)

        if not verified:
            print("\nERROR: 3DGS repository verification failed.")
            return 1

    if args.install:
        print("\nPreparing Python environment...")
        upgrade_pip()

        install_success = install_3dgs_requirements(target)

        if not install_success:
            print(
                "\nDependency installation was incomplete. "
                "See the official 3DGS README for environment-specific "
                "CUDA/PyTorch/native-extension instructions."
            )

    print_manual_build_steps(target)

    print(
        "\nAeroSplat setup script finished successfully.\n"
        "The reference 3DGS repository is ready for integration."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())