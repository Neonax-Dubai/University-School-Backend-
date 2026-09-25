#!/bin/sh
# Container entrypoint for the Zayed inference service.
# The dashboard token is read from the 0600 file written by `manage.py seed_zayed --token-file`
# and exported into THIS process only - it never appears in compose files, env files, image
# layers, `docker inspect` output or process arguments.
set -eu
TOKEN_FILE="${DASHBOARD_TOKEN_FILE:-/app/runtime/dashboard_token}"
if [ -z "${DASHBOARD_TOKEN:-}" ]; then
    if [ ! -r "$TOKEN_FILE" ]; then
        echo "FATAL: $TOKEN_FILE is missing - create it with the dashboard's seed_zayed --token-file" >&2
        exit 2
    fi
    DASHBOARD_TOKEN="$(cat "$TOKEN_FILE")"
    export DASHBOARD_TOKEN
fi
mkdir -p /app/runtime/outbox
exec python3 -u /app/zayed_inference.py
