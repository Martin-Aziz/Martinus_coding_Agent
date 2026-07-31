#!/usr/bin/env bash
# Martinus macOS build helper.
# Usage: ./scripts/build-macos.sh [--no-bundle]
set -euo pipefail

NO_BUNDLE=0
for arg in "$@"; do
  case "$arg" in
    --no-bundle) NO_BUNDLE=1 ;;
    *) echo "Unknown argument: $arg" >&2; exit 1 ;;
  esac
done

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command '$1'. Install it and run this script again." >&2
    exit 1
  fi
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ICONS_DIR="$PROJECT_ROOT/src-tauri/icons"
SVG_ICON="$ICONS_DIR/martinus.svg"
ICNS_ICON="$ICONS_DIR/icon.icns"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This script is intended to run on macOS." >&2
  exit 1
fi

echo "Martinus macOS build helper"
echo "Project root: $PROJECT_ROOT"

require_command node
require_command npm
require_command cargo
require_command rustup
require_command npx

cd "$PROJECT_ROOT"

if [[ ! -d "$PROJECT_ROOT/node_modules" ]]; then
  echo "Installing desktop dependencies..."
  npm install
fi

if [[ ! -f "$ICNS_ICON" ]]; then
  if [[ ! -f "$SVG_ICON" ]]; then
    echo "Cannot generate macOS icon because '$SVG_ICON' is missing." >&2
    exit 1
  fi
  echo "Generating macOS icon assets..."
  npx tauri icon "$SVG_ICON" --output "$ICONS_DIR"
fi

TARGET="$(uname -m)"
if [[ "$TARGET" == "arm64" ]]; then
  RUST_TARGET="aarch64-apple-darwin"
else
  RUST_TARGET="x86_64-apple-darwin"
fi

echo "Ensuring the macOS Rust target ($RUST_TARGET) is installed..."
rustup target add "$RUST_TARGET"

if [[ "$NO_BUNDLE" -eq 1 ]]; then
  echo "Running an unbundled macOS-targeted desktop build..."
  npm run tauri:build -- --target "$RUST_TARGET" --no-bundle
else
  echo "Running the macOS app bundle build (.app + .dmg)..."
  npm run tauri:build -- --target "$RUST_TARGET" --bundles app,dmg
fi

echo ""
echo "Build completed."
echo "Artifacts: $PROJECT_ROOT/src-tauri/target/$RUST_TARGET/release/bundle"
