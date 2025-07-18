#!/bin/bash
# Wrapper script to run dispatching rules evaluation

cd "$(dirname "$0")"

# Create temporary Python script to avoid argparse conflicts
cat > temp_eval_dispatch.py << 'EOF'
import os
import sys

# Get arguments before any imports
max_instances = None
dataset_type = 'all'

for i, arg in enumerate(sys.argv):
    if arg == '--max_instances' and i + 1 < len(sys.argv):
        max_instances = int(sys.argv[i + 1])
    elif arg == '--dataset_type' and i + 1 < len(sys.argv):
        dataset_type = sys.argv[i + 1]

# Now import and run
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
exec(open('evaluate_dispatching_rules.py').read())
EOF

# Run with clean environment
python3 temp_eval_dispatch.py "$@"

# Cleanup
rm -f temp_eval_dispatch.py