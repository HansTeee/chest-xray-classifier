#!/usr/bin/env bash
# Start FastAPI (internal :8000) and Streamlit (public :7860).
# If either process exits, stop the container so the platform restarts it.
set -euo pipefail

echo "Starting FastAPI on :8000 ..."
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1 &
API_PID=$!

# Wait for the API to come up (the model may be downloading from MODEL_URL)
for _ in $(seq 1 120); do
  if python -c "import urllib.request as u; u.urlopen('http://localhost:8000/health', timeout=2)" 2>/dev/null; then
    echo "API is up."
    break
  fi
  if ! kill -0 "$API_PID" 2>/dev/null; then
    echo "API process exited during startup." >&2
    exit 1
  fi
  sleep 1
done

echo "Starting Streamlit on :7860 ..."
streamlit run ui/app.py --server.port 7860 --server.address 0.0.0.0 &
UI_PID=$!

trap 'kill -TERM "$API_PID" "$UI_PID" 2>/dev/null || true' TERM INT
wait -n "$API_PID" "$UI_PID"
STATUS=$?
echo "A process exited (status $STATUS); shutting down." >&2
kill -TERM "$API_PID" "$UI_PID" 2>/dev/null || true
exit "$STATUS"
