#!/usr/bin/env bash

C_FLAG="Y"
B_FLAG="Y"
INDEX_NAME=""
SCRIPT_NAME="$(basename "$0")"

usage() {
    echo "Usage: $SCRIPT_NAME [-c Y|N] [-b Y|N] -i INDEX_NAME"
    echo "  -c    Run correctness tests (Y/N). Default: Y"
    echo "  -b    Run benchmark tests (Y/N). Default: Y"
    echo "  -i    Index name to test (required)"
    echo "  -h    Show this help message"
}

while getopts "c:b:i:h" opt; do
    case "$opt" in
        c)
            C_FLAG="$OPTARG"
            ;;
        b)
            B_FLAG="$OPTARG"
            ;;
        i)
            INDEX_NAME="$OPTARG"
            ;;
        h)
            usage
            exit 0
            ;;
        *)
            usage
            exit 1
            ;;
    esac
done

if [ -z "$INDEX_NAME" ]; then
    echo "Error: -i INDEX_NAME is required"
    usage
    exit 1
fi

case "$C_FLAG" in
    Y|y|N|n) ;;
    *)
        echo "Error: -c must be Y or N"
        usage
        exit 1
        ;;
esac

case "$B_FLAG" in
    Y|y|N|n) ;;
    *)
        echo "Error: -b must be Y or N"
        usage
        exit 1
        ;;
esac

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

python - <<EOF
import sys, os
sys.path.insert(0, os.path.join("$ROOT_DIR", "tests"))
from utils import getIndex
idx = getIndex("$INDEX_NAME", 2, 2)
if not idx:
    sys.stderr.write(f"Error: Invalid index name '$INDEX_NAME'\\n")
    sys.exit(1)
sys.exit(0)
EOF

if [ $? -ne 0 ]; then
    exit 1
fi

if [[ "$C_FLAG" == "Y" || "$C_FLAG" == "y" ]]; then
    echo "-----STARTING CORRECTNESS TESTS-------"
    python "$ROOT_DIR/tests/correct.py" -i "$INDEX_NAME"
    if [ $? -ne 0 ]; then
        echo "Correctness tests failed"
        exit 1
    fi
fi

if [[ "$B_FLAG" == "Y" || "$B_FLAG" == "y" ]]; then
    echo "-----STARTING BENCHMARK TESTS-------"
    python "$ROOT_DIR/tests/bench.py" -i "$INDEX_NAME"
    if [ $? -ne 0 ]; then
        echo "Benchmark tests failed"
        exit 1
    fi
fi
