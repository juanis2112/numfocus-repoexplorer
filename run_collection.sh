#!/bin/bash
# NumFocus Data Collection Runner
# --------------------------------
# Open Terminal, navigate to this folder, and run:
#   bash run_collection.sh
#
# Requires: Python 3.8+, pip packages (auto-installed below)

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== NumFocus Data Collector ==="
echo "Working directory: $SCRIPT_DIR"

# Load GITHUB_TOKEN from .env if present
if [ -f .env ]; then
    export $(grep -v '^#' .env | grep GITHUB_TOKEN | sed 's/ //g' | xargs)
    echo "Loaded GITHUB_TOKEN from .env"
fi

if [ -z "$GITHUB_TOKEN" ]; then
    echo ""
    echo "WARNING: No GITHUB_TOKEN found. The GitHub API will be rate-limited to 60 req/hr."
    echo "To set a token: export GITHUB_TOKEN=your_token_here"
    echo ""
fi

# Install dependencies
echo "Installing Python dependencies..."
pip3 install requests pandas pyarrow --quiet --break-system-packages 2>/dev/null || \
pip3 install requests pandas pyarrow --quiet

echo "Starting data collection (this takes ~3-5 minutes for 63 projects)..."
python3 collect_data.py

echo ""
echo "Done! The parquet file is at Data/parquet/numfocus_projects.parquet"
echo "You can now run the Shiny app: shiny run app.py"
