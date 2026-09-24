"""Download the immutable, checksum-pinned original ONNX. Never request credentials."""
import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess
from urllib.error import URLError
from urllib.request import urlopen
from .model import BASE_URL, BASE_SHA256

BASE_GIT_BLOB = "dadf614b35eb26b3a61db8961ca3a151605fb55f"
MAX_BYTES = 10*1024*1024


def download_base() -> bytes:
    try:
        with urlopen(BASE_URL, timeout=60) as response:
            return response.read(MAX_BYTES+1)
    except URLError as exc:
        # Some sandboxes deny raw.githubusercontent.com while their configured
        # GitHub CLI connection can still access the repository. Do not weaken TLS.
        if shutil.which("gh") is None:
            raise RuntimeError("Direct HTTPS download failed. Install/configure GitHub CLI, or download the pinned file yourself; SHA256 will still be checked.") from exc
        result = subprocess.run([
            "gh", "api", f"repos/Klischa/astra2/git/blobs/{BASE_GIT_BLOB}",
            "-H", "Accept: application/vnd.github.raw+json",
        ], capture_output=True, timeout=60, check=False)
        if result.returncode:
            raise RuntimeError("GitHub CLI could not fetch the pinned model. Check the GitHub connection (in Arena, reconnect GitHub). No credential should be added to this project.") from exc
        return result.stdout


def fetch_base(output: Path) -> str:
    output = Path(output)
    data = output.read_bytes() if output.exists() else download_base()
    if len(data) > MAX_BYTES or hashlib.sha256(data).hexdigest() != BASE_SHA256:
        raise RuntimeError("Base model checksum mismatch; refusing to use or overwrite it")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(data)
    return BASE_SHA256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("weights/original_pose_regressor.onnx"))
    args = parser.parse_args()
    sha = fetch_base(args.output)
    print(f"Verified {args.output}: {sha}")
    print("Upstream weights: Klischa/astra2; CC BY-NC-SA 4.0. See THIRD_PARTY.md.")


if __name__ == "__main__":
    main()
