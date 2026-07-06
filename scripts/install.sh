#!/bin/sh
# Install the inferlab binary (x86_64 / aarch64 Linux) from GitHub releases,
# and optionally unpack the agent plugin package next to it.
#
# Options:
#   --bin-dir <path>   Install the binary to this directory (default: ~/.local/bin)
#   --no-plugin        Skip downloading the agent plugin package
#   --help             Show this help and exit
#
# Environment:
#   INFERLAB_VERSION   Pin a release tag (default: latest)
#   INFERLAB_BASE_URL  Override the release-asset download base URL

set -eu

VERSION="${INFERLAB_VERSION:-latest}"
BIN_DIR="${HOME}/.local/bin"
WITH_PLUGIN=true
REPO="Infer-Lab/inferlab"

show_help() {
    sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
    exit 0
}

die() {
    printf 'install.sh: %s\n' "$1" >&2
    exit 1
}

command -v curl > /dev/null 2>&1 || die "curl is required"

while [ $# -gt 0 ]; do
    case "$1" in
        --bin-dir)   [ $# -ge 2 ] || die "--bin-dir requires a value"
                     BIN_DIR="$2"; shift 2 ;;
        --no-plugin) WITH_PLUGIN=false; shift ;;
        --help|-h)   show_help ;;
        *) die "unknown option: $1" ;;
    esac
done

case "$(uname -sm)" in
    "Linux x86_64")  ASSET="inferlab-x86_64-linux" ;;
    "Linux aarch64") ASSET="inferlab-aarch64-linux" ;;
    *) die "unsupported platform: $(uname -sm) (x86_64/aarch64 Linux only)" ;;
esac

if [ "$VERSION" = latest ]; then
    BASE="${INFERLAB_BASE_URL:-https://github.com/${REPO}/releases/latest/download}"
else
    BASE="${INFERLAB_BASE_URL:-https://github.com/${REPO}/releases/download/${VERSION}}"
fi

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

fetch() {
    curl -fsSL "$BASE/$1" -o "$WORK/$1" || die "failed to download $BASE/$1"
}

fetch "$ASSET"
fetch "$ASSET.sha256"
(cd "$WORK" && sha256sum -c "$ASSET.sha256" >/dev/null) || die "checksum mismatch for $ASSET"

mkdir -p "$BIN_DIR"
install -m 0755 "$WORK/$ASSET" "$BIN_DIR/inferlab"
printf 'installed %s -> %s/inferlab\n' "$ASSET" "$BIN_DIR"

if [ "$WITH_PLUGIN" = true ]; then
    fetch inferlab-plugin.tar.gz
    fetch inferlab-plugin.tar.gz.sha256
    (cd "$WORK" && sha256sum -c inferlab-plugin.tar.gz.sha256 >/dev/null) \
        || die "checksum mismatch for inferlab-plugin.tar.gz"
    PLUGIN_DIR="${HOME}/.local/share/inferlab/plugin"
    rm -rf "$PLUGIN_DIR"
    mkdir -p "$PLUGIN_DIR"
    tar -xzf "$WORK/inferlab-plugin.tar.gz" -C "$PLUGIN_DIR"
    printf 'plugin package -> %s\n' "$PLUGIN_DIR"
    printf 'next: %s/inferlab agent install --agent all --from-checkout %s\n' \
        "$BIN_DIR" "$PLUGIN_DIR"
fi

case ":$PATH:" in
    *":$BIN_DIR:"*) ;;
    *) printf 'note: %s is not on PATH\n' "$BIN_DIR" ;;
esac
