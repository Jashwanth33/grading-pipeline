#!/bin/bash
set -e

echo "=== Installing Python dependencies ==="
pip install -r requirements.txt

echo "=== Installing Node.js dependencies ==="
cd frontend
npm install --legacy-peer-deps

echo "=== Building React frontend ==="
CI=false npm run build

echo "=== Verifying build output ==="
ls -la build/
ls -la build/static/

cd ..

echo "=== Generating sample data ==="
python -m scripts.generate_data

echo "=== Build complete ==="
echo "Frontend build at: $(pwd)/frontend/build"
