#!/usr/bin/env bash
set -e

# =====================
# Usage
# =====================
usage() {
  echo "Usage: $0 <command> [options]"
  echo
  echo "Commands:"
  echo "  register    Register a model from an MLflow run"
  echo "  set-alias   Set an alias for a model version"
  echo "  promote     Promote a model to production"
  echo "  list        List all registered models"
  echo "  info        Get info about a model"
  echo
  echo "Options:"
  echo "  --config PATH       Path to config YAML (default: config/config.yaml)"
  echo "  --run-id ID         MLflow run ID (for register)"
  echo "  --model-name NAME   Model name"
  echo "  --version VER       Model version (for set-alias, promote)"
  echo "  --alias ALIAS       Alias to set: staging|champion|production (for set-alias)"
  echo "  --description DESC  Model description (for register)"
  echo "  -h, --help          Show this help message"
  echo
  echo "Examples:"
  echo "  $0 register --run-id abc123 --model-name churn-model"
  echo "  $0 set-alias --model-name churn-model --version 1 --alias staging"
  echo "  $0 promote --model-name churn-model --version 1"
  echo "  $0 list"
  echo "  $0 info --model-name churn-model"
  exit 1
}

# =====================
# Resolve paths
# =====================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# =====================
# Defaults
# =====================
PYTHON_SCRIPT="$PROJECT_ROOT/src/scripts/register_model.py"
CONFIG_PATH="$PROJECT_ROOT/src/config/config.yaml"

# =====================
# Python path
# =====================
export PYTHONPATH="$PROJECT_ROOT"

# =====================
# Parse --config from arguments
# =====================
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    -h|--help)
      usage
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ ${#ARGS[@]} -eq 0 ]]; then
  usage
fi

# =====================
# Run
# =====================
python "$PYTHON_SCRIPT" --config "$CONFIG_PATH" "${ARGS[@]}"
