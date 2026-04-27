import subprocess
import sys
import argparse
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the full route scoring pipeline.')
    parser.add_argument('--base_dir', type=Path,
                        default=Path(__file__).resolve().parent.parent / 'outputs_revision' / 'configuration_0')
    args = parser.parse_args()

    subprocess.run([sys.executable, SCRIPTS_DIR / 'prepare_full_routes.py', '--base_dir', args.base_dir], check=True)

    subprocess.run([sys.executable, SCRIPTS_DIR / 'score_full_routes.py', '--base_dir', args.base_dir], check=True)

    subprocess.run([sys.executable, SCRIPTS_DIR / 'classify_routes_risk.py', '--base_dir', args.base_dir], check=True)
