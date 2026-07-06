#!/bin/sh
# Pack the agent plugin package reproducibly: sorted member order, fixed
# owner and mtime, and a gzip header without name or timestamp, so the
# same content hashes identically across filesystems and machines. Used
# by both `just plugin-tarball` and the release workflow.
set -eu

OUT="${1:?usage: pack-plugin.sh <out.tar.gz>}"

tar --sort=name \
    --owner=root --group=root --numeric-owner \
    --mtime='2026-01-01 00:00:00 UTC' \
    -cf - LICENSE .claude-plugin/ .agents/ plugins/ | gzip -n > "$OUT"

# License retention (RFC-0001:C-LICENSE-RETENTION): the plugin package packs
# the notice, asserted here.
tar -tzf "$OUT" | grep -q '^LICENSE$'
tar -tzf "$OUT" | grep -q '^plugins/inferlab/skills/inferlab/SKILL.md$'
tar -tzf "$OUT" | grep -q '^.claude-plugin/marketplace.json$'
tar -tzf "$OUT" | grep -q '^.agents/plugins/marketplace.json$'
