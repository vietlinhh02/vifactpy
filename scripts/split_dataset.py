#!/usr/bin/env python
import os
import sys

# Allow running from scripts/ by adding project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vifact.cli import main


if __name__ == "__main__":
    # Usage: python scripts/split_dataset.py --input ... --output ...
    main(["split", *sys.argv[1:]])
