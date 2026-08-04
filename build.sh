#!/bin/bash
set -e

echo "=== Installing Python dependencies ==="
pip install -r requirements.txt

echo "=== Installing Node.js dependencies ==="
cd frontend
npm install --legacy-peer-deps

echo "=== Building React frontend ==="
npx react-scripts build

echo "=== Generating sample data ==="
cd ..
python -m scripts.generate_data

echo "=== Build complete ==="
