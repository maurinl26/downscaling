#!/usr/bin/env python
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "downscaling[pmap]",
# ]
# ///
"""
Forwarder vers downscaling.scripts.run_campaign.

Pour un usage sans installation préalable :
    uv run runs/scripts/run_campaign.py --source era5land --step all

Pour un usage après installation (uv sync --extra pmap) :
    run-campaign --source era5land --step all
"""
from downscaling.scripts.run_campaign import main

if __name__ == "__main__":
    main()
