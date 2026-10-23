"""Shared release steps; consumer workflow configuration supplies package policy."""

import argparse
import ast
import fnmatch
import json
import os
import re
import subprocess
import urllib.request
import zipfile
from pathlib import Path


def checked_path(root, name):
    path = (root / name).resolve()
    path.relative_to(root.resolve())
    return path


def verify(root, config, version):
    if not re.fullmatch(
        r"[0-9]+\.[0-9]+\.[0-9]+(?:a[0-9]+|b[0-9]+|rc[0-9]+)?", version
    ):
        raise ValueError("Invalid release version")
    tree = ast.parse(checked_path(root, config["source"]).read_text(encoding="utf-8"))
    values = [
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "PLUGIN_VERSION"
            for target in node.targets
        )
    ]
    if values != [version]:
        raise ValueError(
            f"Requested version {version} does not match PLUGIN_VERSION: {values}"
        )


def package(root, config, version):
    verify(root, config, version)
    package_name = config["package"]
    source = checked_path(root, package_name)
    output = checked_path(root, config["archive"].format(version=version))
    if output == source or source in output.parents:
        raise ValueError("Release archive must be outside the package")
    excluded = [
        "*/__pycache__/*",
        "*/.git/*",
        "*/settings.json",
        "*.pyc",
        *config.get("exclude", []),
    ]
    entries = {}
    for path in source.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if path.is_file() and not any(
            fnmatch.fnmatch(relative, pattern) for pattern in excluded
        ):
            if path.is_symlink():
                raise ValueError(f"Symlinks are not release inputs: {path}")
            entries[relative] = path
    for name in config.get(
        "include", ["README.md", "LICENSE", "COPYING", "LICENSING.md"]
    ):
        path = checked_path(root, name)
        if not path.exists():
            raise FileNotFoundError(path)
        paths = sorted(path.rglob("*")) if path.is_dir() else [path]
        for item in paths:
            if item.is_file():
                archive_name = (
                    str(Path(package_name) / name / item.relative_to(path))
                    if path.is_dir()
                    else str(Path(package_name) / name)
                )
                entries[Path(archive_name).as_posix()] = item
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, path in sorted(entries.items()):
            entry = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            entry.external_attr = 0o100644 << 16
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, path.read_bytes())
    print(output.name)
    return output


def publish(root, config, version):
    verify(root, config, version)
    assets = [name.format(version=version) for name in config["assets"]]
    for name in assets:
        if not checked_path(root, name).is_file():
            raise FileNotFoundError(name)
    tag = "v" + version
    result = subprocess.run(
        ["gh", "release", "view", tag],
        cwd=root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode == 0:
        command = ["gh", "release", "upload", tag, *assets, "--clobber"]
    else:
        command = [
            "gh",
            "release",
            "create",
            tag,
            "--verify-tag",
            "--title",
            config["title"] + " " + version,
            "--generate-notes",
            *assets,
        ]
    subprocess.run(command, cwd=root, check=True)


def notify(version):
    token = os.environ.get("REGISTRY_PAT", "")
    if not token:
        print(
            "No REGISTRY_PAT: register the release through the plugin collection issue form."
        )
        registered = False
    else:
        data = json.dumps(
            {
                "event_type": "plugin_release",
                "client_payload": {
                    "repo": os.environ["GITHUB_REPOSITORY"],
                    "tag": "v" + version,
                },
            }
        ).encode()
        request = urllib.request.Request(
            "https://api.github.com/repos/HiroYokoyama/moleditpy-plugins/dispatches",
            data=data,
            headers={
                "Authorization": "Bearer " + token,
                "Accept": "application/vnd.github+json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=30):
            pass
        registered = True
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as handle:
            handle.write(f"registered={str(registered).lower()}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("verify", "package", "publish", "notify"))
    parser.add_argument("--version", required=True)
    parser.add_argument("--config", default="scripts/release-config.json")
    args = parser.parse_args()
    root = Path.cwd()
    if args.step == "notify":
        notify(args.version)
    else:
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        {"verify": verify, "package": package, "publish": publish}[args.step](
            root, config, args.version
        )


if __name__ == "__main__":
    main()
