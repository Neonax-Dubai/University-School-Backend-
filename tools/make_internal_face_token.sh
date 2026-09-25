#!/usr/bin/env bash
# Create the internal face API token (0600) if it does not exist. Never printed, never in git.
set -euo pipefail
FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/runtime/internal_face_token"
if [ -s "$FILE" ]; then
    echo "internal face token already present at $FILE (unchanged)"
    exit 0
fi
install -m 700 -d "$(dirname "$FILE")"
umask 077
python3 -c "import secrets; print(secrets.token_urlsafe(48))" > "$FILE"
chmod 600 "$FILE"
echo "internal face token created at $FILE (0600, not shown)"
