#!/usr/bin/env python3
"""Read and validate the package metadata used by a Python package release."""

from __future__ import annotations

import pathlib
import sys
import tomllib
from typing import NoReturn

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

ROOT = pathlib.Path(__file__).resolve().parent.parent
SDK_DISTRIBUTION = "inferlab-adapter-sdk"


def fail(message: str) -> NoReturn:
    raise SystemExit(f"prepare-python-package-release: {message}")


def project_table(path: pathlib.Path) -> dict[str, object]:
    with path.open("rb") as source:
        document = tomllib.load(source)
    project = document.get("project")
    if not isinstance(project, dict):
        fail(f"{path.relative_to(ROOT)}: no [project] table found")
    return project


def string_field(project: dict[str, object], field: str, path: pathlib.Path) -> str:
    value = project.get(field)
    if not isinstance(value, str) or not value:
        fail(f"{path.relative_to(ROOT)}: no project {field} found")
    return value


def validate_sdk_dependency(
    project: dict[str, object], path: pathlib.Path, sdk_version: str
) -> None:
    raw_dependencies = project.get("dependencies")
    if not isinstance(raw_dependencies, list) or not all(
        isinstance(dependency, str) for dependency in raw_dependencies
    ):
        fail(f"{path.relative_to(ROOT)}: project dependencies must be a string list")

    sdk_requirements: list[Requirement] = []
    for raw_dependency in raw_dependencies:
        try:
            requirement = Requirement(raw_dependency)
        except InvalidRequirement as error:
            fail(f"{path.relative_to(ROOT)}: invalid runtime dependency: {error}")
        if canonicalize_name(requirement.name) == canonicalize_name(SDK_DISTRIBUTION):
            sdk_requirements.append(requirement)

    exact = False
    requirement_version = ""
    if len(sdk_requirements) == 1:
        requirement = sdk_requirements[0]
        specifiers = list(requirement.specifier)
        if (
            requirement.url is None
            and not requirement.extras
            and requirement.marker is None
            and len(specifiers) == 1
            and specifiers[0].operator == "=="
            and "*" not in specifiers[0].version
        ):
            exact = True
            requirement_version = specifiers[0].version

    if not exact:
        fail(
            f"{path.relative_to(ROOT)}: integration requires exactly one exact "
            f"{SDK_DISTRIBUTION} runtime dependency"
        )

    try:
        versions_agree = Version(requirement_version) == Version(sdk_version)
    except InvalidVersion as error:
        fail(f"{path.relative_to(ROOT)}: invalid SDK version: {error}")
    if not versions_agree:
        fail(
            f"{path.relative_to(ROOT)}: SDK dependency {requirement_version} "
            f"!= checkout SDK {sdk_version}"
        )


def main() -> None:
    if len(sys.argv) != 2:
        fail(f"usage: {pathlib.Path(sys.argv[0]).name} PACKAGE")

    package = sys.argv[1]
    path = ROOT / "python" / package / "pyproject.toml"
    project = project_table(path)
    distribution = string_field(project, "name", path)
    version = string_field(project, "version", path)

    if package.startswith("inferlab-integration-"):
        sdk_path = ROOT / "python" / SDK_DISTRIBUTION / "pyproject.toml"
        sdk_project = project_table(sdk_path)
        sdk_version = string_field(sdk_project, "version", sdk_path)
        validate_sdk_dependency(project, path, sdk_version)

    print(distribution)
    print(version)


if __name__ == "__main__":
    main()
