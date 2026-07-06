#!/usr/bin/env python3
import hashlib
import json
import os
import sys
import time


def scenario():
    path = os.environ.get("FIXTURE_SCENARIO")
    if not path:
        return {}
    with open(path) as handle:
        return json.load(handle)


fault = scenario()
args = sys.argv[1:]
log = fault.get("docker_log")
if log:
    with open(log, "a") as handle:
        handle.write(" ".join(args) + "\n")
if args[:2] == ["version", "--format"]:
    print("linux/amd64")
elif args[:3] == ["manifest", "inspect", "--verbose"]:
    reference = args[3]

    def digest(platform):
        return "sha256:" + hashlib.sha256(f"{reference}|{platform}".encode()).hexdigest()

    print(
        json.dumps(
            [
                {
                    "Descriptor": {
                        "digest": digest("linux/amd64"),
                        "platform": {"os": "linux", "architecture": "amd64"},
                    }
                },
                {
                    "Descriptor": {
                        "digest": digest("linux/arm64"),
                        "platform": {"os": "linux", "architecture": "arm64"},
                    }
                },
            ]
        )
    )
elif args[0] == "build":
    # The durable record must already carry this exact command while it is
    # still executing (record-before-first-external-effect).
    record_path = os.path.join(os.path.dirname(args[-1]), "record.json")
    with open(record_path) as handle:
        record = json.load(handle)
    recorded = [
        command["argv"]
        for assembly in record["assemblies"]
        for command in assembly["native_commands"]
    ]
    if ["docker", *args] not in recorded:
        sys.stderr.write("docker build command was not persisted before execution\n")
        sys.exit(3)
    iidfile = args[args.index("--iidfile") + 1]
    with open(os.path.join(args[-1], "Dockerfile"), "rb") as dockerfile:
        content = dockerfile.read()
    with open(iidfile, "w") as handle:
        handle.write("sha256:" + hashlib.sha256(content).hexdigest())
elif args[:2] == ["image", "inspect"]:
    if "{{.Id}}" in args:
        # The external-image presence probe; absence can be global or scoped
        # to one fixture machine via the ssh shim's target marker.
        absent_on = fault.get("external_absent_on_target")
        if fault.get("external_absent") or (
            absent_on and absent_on == os.environ.get("FIXTURE_SSH_TARGET")
        ):
            sys.stderr.write("Error: No such image: " + args[-1] + "\n")
            sys.exit(1)
        print("sha256:fixtureexternalid")
    else:
        print(json.dumps(["/usr/local/bin/inferlab-entrypoint"]))
elif args[0] == "save":
    with open(args[args.index("--output") + 1], "w") as handle:
        handle.write("oci archive for " + args[-1] + "\n")
elif args[0] == "rm":
    if fault.get("rm_fail"):
        # A non-zero docker rm that is not an absent container: the
        # structured removal reason must carry this exit, distinct from a
        # deadline.
        sys.stderr.write("Error response from daemon: cannot remove running container\n")
        sys.exit(1)
    # logged above; the fixture has nothing to remove
elif args[0] == "run":
    image_index = next(
        index for index, arg in enumerate(args) if arg.startswith("sha256:") or "@sha256:" in arg
    )
    inner = args[image_index + 1 :]
    if "--entrypoint" in args[:image_index]:
        inner = [args[args.index("--entrypoint") + 1], *inner]
    if "--cidfile" in args:
        with open(args[args.index("--cidfile") + 1], "w") as handle:
            handle.write("fixturecid0123")
    if inner[0] in ("python", "python3") and inner[1] == "-c" and "importlib.metadata" in inner[2]:
        # The external-image framework version probe.
        print("0.7.fixture")
        sys.exit(0)
    if (
        inner[0] in ("python", "python3")
        and inner[1] == "-m"
        and inner[2].startswith("inferlab_integration_")
    ):
        if fault.get("adapter_hang"):
            time.sleep(60)
            sys.exit(3)
        # The image's serving stack answers integration requests; the
        # fixture's stack is the fixture adapter.
        adapter = "inferlab-adapter-" + inner[2].removeprefix("inferlab_integration_")
        os.execvp(adapter, [adapter, *inner[3:]])
    os.execvp(inner[0], inner)
else:
    sys.stderr.write(f"unexpected docker fixture arguments: {args!r}\n")
    sys.exit(2)
