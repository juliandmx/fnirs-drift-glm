"""Laufprovenienz: Metadaten mit Code-, Daten- und Konfigurationsfingerabdruck, Archiv
und atomare Schreibvorgaenge fuer Ergebnisdateien.

Jeder Analyselauf schreibt neben seiner CSV eine `<name>.meta.json` mit Git-Commit und
-Status des Repos, Cedalion-Commit, SHA-256 aller Quelldateien des Pakets, SHA-256 der
Eingabedateien und der Konfiguration. `config_fingerprint` haengt nur von Konfiguration,
Quellcode, Eingabedaten und Cedalion-Commit ab (nicht vom Git-Status), damit ein Resume
nur dann fortsetzt, wenn nichts davon veraendert wurde. Ergebnisdateien werden erst in
eine temporaere Datei geschrieben und dann per `os.replace` ersetzt; vor dem Ersetzen
einer vorhandenen Datei legt `archive_file` eine verifizierte Kopie unter
`results/archive/` ab.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from drift_glm import paths

PACKAGE = paths.ROOT / "drift_glm"
ARCHIVE = paths.RESULTS / "archive"


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        return subprocess.check_output(["git", *args], cwd=cwd,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def source_hashes(package: Path = PACKAGE) -> dict[str, str]:
    """SHA-256 aller Python-Quelldateien des Pakets, Schluessel relativ zur Repo-Wurzel."""
    return {str(p.relative_to(paths.ROOT)).replace(os.sep, "/"): sha256_file(p)
            for p in sorted(package.rglob("*.py")) if "__pycache__" not in p.parts}


def cedalion_commit() -> str | None:
    """Commit des eingebundenen Cedalion-Clones (editable install), sonst None."""
    try:
        import cedalion
    except ImportError:                                        # pragma: no cover
        return None
    root = Path(cedalion.__file__).resolve()
    for parent in root.parents:
        if (parent / ".git").exists():
            return _git(["rev-parse", "HEAD"], parent)
    return None


def _jsonable(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (set, tuple)):
        return list(obj)
    if hasattr(obj, "item"):                                   # numpy scalars
        return obj.item()
    return str(obj)


def build_run_metadata(config: dict, data_files=()) -> dict:
    """Metadaten eines Laufs; `config_fingerprint` identifiziert Code + Daten + Config."""
    data = []
    for p in data_files:
        p = Path(p)
        data.append(dict(path=str(p.resolve()), bytes=p.stat().st_size,
                         sha256=sha256_file(p)))
    src = source_hashes()
    ced = cedalion_commit()
    fingerprint_source = dict(config=config, source_sha256=src,
                              data_sha256=[d["sha256"] for d in data],
                              cedalion_commit=ced)
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_source, sort_keys=True, default=_jsonable).encode()
    ).hexdigest()
    return dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        git_commit=_git(["rev-parse", "HEAD"], paths.ROOT),
        git_status=_git(["status", "--porcelain"], paths.ROOT),
        cedalion_commit=ced,
        config=json.loads(json.dumps(config, default=_jsonable)),
        data_files=data,
        source_sha256=src,
        config_fingerprint=fingerprint,
    )


def archive_file(path: str | Path, reason: str) -> Path | None:
    """Verifizierte Kopie einer Ergebnisdatei nach results/archive/, Eintrag im Index.

    Rueckgabe: Pfad der Kopie, oder None, wenn `path` nicht existiert.
    """
    src = Path(path)
    if not src.exists():
        return None
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S_%f")
    safe_reason = "".join(c if c.isalnum() or c in "-_." else "_" for c in reason)
    target = ARCHIVE / f"{src.stem}_{stamp}_{safe_reason}{src.suffix}"
    digest = sha256_file(src)
    target.write_bytes(src.read_bytes())
    if sha256_file(target) != digest:
        raise IOError(f"Archivkopie stimmt nicht mit dem Original ueberein: {src}")
    with open(ARCHIVE / "index.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),
                                 source=str(src), archive=str(target),
                                 sha256=digest, reason=reason)) + "\n")
    return target


def _replace_atomic(write, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def write_csv_atomic(frame: pd.DataFrame, path: str | Path) -> None:
    _replace_atomic(lambda tmp: frame.to_csv(tmp, index=False), Path(path))


def write_metadata_atomic(meta: dict, path: str | Path) -> None:
    _replace_atomic(lambda tmp: tmp.write_text(
        json.dumps(meta, indent=2, default=_jsonable) + "\n", encoding="utf-8"),
        Path(path))


write_json_atomic = write_metadata_atomic
