"""Fingerprint, archive and atomic writes of the provenance helpers."""
import json

import pandas as pd
import pytest

from drift_glm.core import provenance as pv


def test_fingerprint_depends_on_config_but_not_on_git_status(tmp_path):
    data = tmp_path / "in.txt"
    data.write_text("abc")
    a = pv.build_run_metadata({"x": 1, "cells": [("a", "b")]}, [data])
    b = pv.build_run_metadata({"x": 1, "cells": [("a", "b")]}, [data])
    c = pv.build_run_metadata({"x": 2, "cells": [("a", "b")]}, [data])
    assert a["config_fingerprint"] == b["config_fingerprint"]
    assert a["config_fingerprint"] != c["config_fingerprint"]
    assert a["data_files"][0]["sha256"] == pv.sha256_file(data)
    assert "drift_glm/core/provenance.py" in a["source_sha256"]
    json.dumps(a)                       # serialisable as written


def test_archive_and_atomic_writes(tmp_path, monkeypatch):
    monkeypatch.setattr(pv, "ARCHIVE", tmp_path / "archive")
    out = tmp_path / "res.csv"
    pv.write_csv_atomic(pd.DataFrame(dict(a=[1, 2])), out)
    assert pd.read_csv(out).a.tolist() == [1, 2]
    assert not list(tmp_path.glob(".res.csv.*.tmp"))
    copy = pv.archive_file(out, "unit test/reason")
    assert copy.exists() and copy.read_bytes() == out.read_bytes()
    assert copy.name.endswith("unit_test_reason.csv")
    index = (tmp_path / "archive" / "index.jsonl").read_text().strip().splitlines()
    assert json.loads(index[-1])["sha256"] == pv.sha256_file(out)
    assert pv.archive_file(tmp_path / "missing.csv", "x") is None
    meta = tmp_path / "res.meta.json"
    pv.write_metadata_atomic({"k": (1, 2)}, meta)
    assert json.loads(meta.read_text()) == {"k": [1, 2]}
