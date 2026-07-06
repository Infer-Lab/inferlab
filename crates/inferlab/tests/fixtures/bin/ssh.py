#!/usr/bin/env python3
import json
import os
import subprocess
import sys


def scenario():
    path = os.environ.get("FIXTURE_SCENARIO")
    if not path:
        return {}
    with open(path) as handle:
        return json.load(handle)


fault = scenario()
args = sys.argv[1:]
index = 0
while index < len(args) and args[index] == "-o":
    index += 2
if index < len(args) and args[index] == "--":
    index += 1
target = args[index]
command = " ".join(args[index + 1 :]).replace("bash -lic", "bash -c", 1)
env = dict(os.environ, FIXTURE_SSH_TARGET=target)
if fault.get("ssh_hang_rm") and "docker rm -f" in command:
    # A live connection to a wedged remote daemon: never returns.
    import time

    time.sleep(3600)
if fault.get("ssh_fail_cleanup") and "read pid expected" in command:
    # The incomplete-launch cleanup script (its handle read is distinctive)
    # fails while the connection is fine.
    sys.stderr.write("fixture: forced launcher cleanup failure\n")
    sys.exit(6)
if fault.get("ssh_swallow_handle"):
    # Deliver everything except the launch handle, as a broken transport
    # or interposed shell might.
    result = subprocess.run(["sh", "-c", command], env=env, capture_output=True)
    kept = [line for line in result.stdout.splitlines() if b"INFERLAB_HANDLE" not in line]
    sys.stdout.buffer.write(b"\n".join(kept) + b"\n")
    sys.stderr.buffer.write(result.stderr)
    sys.exit(result.returncode)
sys.exit(subprocess.run(["sh", "-c", command], env=env).returncode)
