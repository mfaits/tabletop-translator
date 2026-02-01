#!/usr/bin/env bash
set -e

cd /home/mfaits/translator

# Ensure we can draw on the desktop session
export DISPLAY=:0
export XAUTHORITY=/home/mfaits/.Xauthority

# Activate venv
source /home/mfaits/translator/.venv/bin/activate

# Run the app
exec python /home/mfaits/translator/ui_app.py
