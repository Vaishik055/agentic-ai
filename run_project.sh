#!/bin/bash
set -e

echo "=========================================================="
echo "🌾 Starting Agentic AI Assistant for Sustainable Farming"
echo "=========================================================="

if ! command -v python3 &> /dev/null; then
    echo "❌ Error: python3 is not installed or not in PATH."
    exit 1
fi

echo "📦 Verifying Python dependencies..."
if ! python3 -c "import streamlit" &> /dev/null; then
    echo "⚠️ Streamlit not found in current environment. Installing core packages..."
    python3 -m pip install streamlit fastapi uvicorn requests scikit-learn pandas numpy joblib python-dotenv
fi

echo "🧹 Checking for stale processes on ports 8000 and 8501..."
lsof -ti:8000 | xargs kill -9 2>/dev/null || true
lsof -ti:8501 | xargs kill -9 2>/dev/null || true

echo "🚀 Launching FastAPI Backend on http://127.0.0.1:8000..."
python3 -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 > backend.log 2>&1 &
BACKEND_PID=$!
echo "   Backend running with PID: $BACKEND_PID"

sleep 3

echo "🌐 Launching Streamlit Frontend on http://localhost:8501..."
echo "   Press Ctrl+C to stop both servers."
echo "=========================================================="

trap "echo '🛑 Shutting down services...'; kill $BACKEND_PID 2>/dev/null; exit 0" SIGINT SIGTERM

python3 -m streamlit run frontend/app.py --server.port 8501