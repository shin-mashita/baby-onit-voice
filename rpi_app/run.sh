#!/usr/bin/env bash
# Launch the voice agent dashboard on the Pi: http://<pi-ip>:7860
# --host 0.0.0.0 so it's reachable from other devices on the LAN.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
# shellcheck disable=SC1091
source .venv/bin/activate
python -m voice_ui --host 0.0.0.0
