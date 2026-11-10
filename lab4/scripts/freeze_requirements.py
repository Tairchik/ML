"""Save all installed versions and the runtime-only dependency closure."""
import importlib.metadata as metadata
import subprocess
import sys
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[1]


def main():
    frozen = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    (ROOT / "requirements.txt").write_text("# Full verified Python 3.12 environment\n" + frozen)
    roots = [Requirement(line).name for line in (ROOT / "requirements-runtime.in").read_text().splitlines()
             if line and not line.startswith("#")]
    resolved = {}

    def visit(name):
        normalized = canonicalize_name(name)
        if normalized in resolved:
            return
        package = metadata.distribution(name)
        resolved[normalized] = f"{package.metadata['Name']}=={package.version}"
        for specification in package.requires or []:
            requirement = Requirement(specification)
            if requirement.marker is None or requirement.marker.evaluate({"extra": ""}):
                visit(requirement.name)

    for name in roots:
        visit(name)
    (ROOT / "requirements-runtime.txt").write_text(
        "# Runtime closure, resolved on Ubuntu 24.04 / Python 3.12\n" +
        "\n".join(resolved[name] for name in sorted(resolved)) + "\n")
    print(f"Saved {len(frozen.splitlines())} full and {len(resolved)} runtime dependencies")


if __name__ == "__main__":
    main()
