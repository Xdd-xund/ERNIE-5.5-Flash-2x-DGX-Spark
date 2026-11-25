#!/usr/bin/env python3
"""
Automated Open-Source Release Hygiene & Reproducibility Verifier.

Checks every file in the repository for:
  1. Zero leaked credentials, API tokens, or private keys.
  2. Zero private Tailscale/CGNAT IPs or hardcoded personal home directories.
  3. Zero `file://` local editor links or broken relative Markdown file links.
  4. Valid Python syntax (`ast.parse`) across all `.py` files.
  5. Valid Bash syntax (`bash -n`) and executable permissions across all `.sh` files.
  6. Valid JSON structure across all `.json` files.

Optional environment variable:
  HYGIENE_EXTRA_DENY_REGEX : Additional custom regex pattern to reject during local pre-release checks.
"""
from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}

# Build universal patterns dynamically so this file itself never contains literal forbidden prefixes
_SCHEME_FILE = "file" + ":///"
_PAT_HOME = re.compile(r"(?:/" + r"Users/[A-Za-z0-9._-]+|/" + r"home/(?!ubuntu\b)[A-Za-z0-9._-]+)")
_PAT_CGNAT = re.compile(r"\b100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.\d{1,3}\.\d{1,3}\b")
_PAT_TOKENS = re.compile(
    r"\b(?:ghp_[A-Za-z0-9]{20,}|gho_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,})\b"
)
_PAT_PRIVKEY = re.compile(r"-----BEGIN " + r"(?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----")

FORBIDDEN_REGEXES: list[tuple[str, re.Pattern[str]]] = [
    ("Tailscale CGNAT IP (100.64-127.x.x)", _PAT_CGNAT),
    ("Hardcoded personal home directory", _PAT_HOME),
    ("Local file URI in repository", re.compile(re.escape(_SCHEME_FILE))),
    ("API/Cloud credential token", _PAT_TOKENS),
    ("Private SSH/TLS key header", _PAT_PRIVKEY),
]

_EXTRA_DENY = os.environ.get("HYGIENE_EXTRA_DENY_REGEX", "").strip()
if _EXTRA_DENY:
    FORBIDDEN_REGEXES.append(("Custom HYGIENE_EXTRA_DENY_REGEX match", re.compile(_EXTRA_DENY, re.IGNORECASE)))

MD_LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


def iter_repo_files(root: Path) -> list[Path]:
    out: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for fn in sorted(filenames):
            if fn == ".DS_Store":
                continue
            out.append(Path(dirpath) / fn)
    return out


def verify_repo(root: Path = REPO_ROOT) -> list[str]:
    errors: list[str] = []
    files = iter_repo_files(root)
    if not files:
        return ["Repository appears empty"]

    for path in files:
        rel = path.relative_to(root)
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            errors.append(f"{rel}: unexpected binary / non-UTF-8 file")
            continue

        # 1. PII / Secret scan across ALL files (zero file exemptions needed)
        for label, pat in FORBIDDEN_REGEXES:
            for m in pat.finditer(text):
                line_no = text.count("\n", 0, m.start()) + 1
                errors.append(f"{rel}:{line_no}: forbidden {label}: {m.group(0)!r}")

        # 2. File-type syntax & permission checks
        if path.suffix == ".py":
            try:
                ast.parse(text, filename=str(rel))
            except SyntaxError as exc:
                errors.append(f"{rel}:{exc.lineno}: Python SyntaxError: {exc.msg}")

        elif path.suffix == ".sh":
            if not os.access(path, os.X_OK):
                errors.append(f"{rel}: shell script is missing executable (+x) permission")
            proc = subprocess.run(["bash", "-n", str(path)], text=True, capture_output=True)
            if proc.returncode != 0:
                errors.append(f"{rel}: bash -n failed: {proc.stderr.strip()}")

        elif path.suffix == ".json":
            try:
                json.loads(text)
            except json.JSONDecodeError as exc:
                errors.append(f"{rel}:{exc.lineno}: invalid JSON: {exc.msg}")

        elif path.suffix == ".md":
            stripped = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
            for m in MD_LINK_RE.finditer(stripped):
                target = m.group(1).strip().split()[0]
                if target.startswith(("http://", "https://", "mailto:", "#")):
                    continue
                clean_target = target.split("#", 1)[0]
                if not clean_target:
                    continue
                resolved = (path.parent / clean_target).resolve()
                if not resolved.exists():
                    errors.append(f"{rel}: broken Markdown link target {target!r}")

    return errors


def main() -> None:
    errs = verify_repo(REPO_ROOT)
    if errs:
        print("FAILED release hygiene checks:", file=sys.stderr)
        for e in errs:
            print(f"  - {e}", file=sys.stderr)
        sys.exit(1)
    print(f"[PASS] Verified {len(iter_repo_files(REPO_ROOT))} repository files: zero secrets/PII, valid syntax, valid links.")


if __name__ == "__main__":
    main()
