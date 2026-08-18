#!/usr/bin/env python3
"""Standard Step 13 entry point: market-independent NFL v4 chronological OOF."""
from __future__ import annotations

import sys

from step13_market_independent_v4 import main

if __name__ == "__main__":
    sys.argv = [arg for arg in sys.argv if arg != "--reset"]
    raise SystemExit(main())
