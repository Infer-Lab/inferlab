set shell := ["bash", "-euo", "pipefail", "-c"]

# Operator-private governance and publication recipes (absent in the public
# tree; the public justfile stands alone without them).
import? "./gov/just/private.just"

default:
    @just --list

# ---------------------------------------------------------------------------- #
#                                     BUILD                                    #
# ---------------------------------------------------------------------------- #

# Build the whole workspace
build:
    cargo build --workspace

# Run the full Rust test suite
test:
    cargo test --workspace --all-targets

# ---------------------------------------------------------------------------- #
#                                    CHECKS                                    #
# ---------------------------------------------------------------------------- #

[group("checks")]
fmt:
    cargo fmt --all

[group("checks")]
fmt-check:
    cargo fmt --all --check

[group("checks")]
clippy:
    cargo clippy --workspace --all-targets -- -D warnings

[group("checks")]
doc:
    RUSTDOCFLAGS='-D warnings' cargo doc --workspace --no-deps

# The full Rust gate, exactly what CI's rust job runs
[group("checks")]
verify-rust: fmt-check build test clippy doc

# The full Python gate, exactly what CI's python job runs
[group("checks")]
verify-python:
    pixi run verify-python

# Everything CI runs
[group("checks")]
verify: verify-rust verify-python

# ---------------------------------------------------------------------------- #
#                                    RELEASE                                   #
# ---------------------------------------------------------------------------- #

# Reproducible plugin tarball, via the same script the release workflow runs
[group("release")]
plugin-tarball out="inferlab-plugin.tar.gz":
    scripts/pack-plugin.sh "{{ out }}"
    sha256sum "{{ out }}"

# Install the operator skill into local agent runtimes from this checkout
[group("release")]
install-skill:
    cargo run --quiet -p inferlab -- agent install --agent all --from-checkout .
