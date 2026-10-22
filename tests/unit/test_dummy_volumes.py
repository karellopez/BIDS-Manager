"""Removing the first volumes of a functional run at conversion.

The image loses its first volumes with every stored value and header field
otherwise kept; the sidecar records it (``NumberOfVolumesDiscardedByUser``)
and its ``VolumeTiming`` follows; the events move with the new first volume;
the run's physiology starts there too; a cell in the table wins over the
default, 0 included; only functional time series are touched.
"""

from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from bidsmgr.cli.convert import (
    TaskOptions,
    _build_tasks_for_subject,
    _drop_volumes_for,
)
from bidsmgr.converter.types import ConvertResult, ConvertTask
from bidsmgr.fixups.dummy_volumes import FIELD, applies, drop_leading_volumes


def _bold(folder: Path, volumes: int = 10, tr: float = 0.8) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    data = np.arange(4 * 4 * 3 * volumes, dtype=np.uint16).reshape(4, 4, 3, volumes)
    img = nib.Nifti1Image(data, np.diag([2.0, 2.0, 2.5, 1.0]))
    img.header.set_zooms((2.0, 2.0, 2.5, tr))
    img.header.set_xyzt_units("mm", "sec")
    img.header["descrip"] = b"TE=30;Time=102012.133;phase=1"
    path = folder / "sub-01_task-rest_bold.nii.gz"
    nib.save(img, str(path))
    (folder / "sub-01_task-rest_bold.json").write_text(json.dumps(
        {"RepetitionTime": tr, "TaskName": "rest"}))
    return path


def _result(image: Path, n: int, *, datatype="func", suffix="bold") -> ConvertResult:
    task = ConvertTask(
        row_id="r", series_uid="1.2", source_files=(image,), dataset="d",
        bids_root=image.parent, subject="01", datatype=datatype, suffix=suffix,
        basename=image.name.split(".")[0], drop_volumes=n)
    return ConvertResult(task=task, success=True,
                         staged_files=(image, image.with_name(image.name.split(".")[0] + ".json")))


def test_the_first_volumes_go_and_everything_else_stays(tmp_path) -> None:
    image = _bold(tmp_path / "func")
    before = nib.load(str(image))
    raw = np.asanyarray(before.dataobj)
    assert drop_leading_volumes([_result(image, 3)]) == 1
    after = nib.load(str(image))
    assert after.shape == (4, 4, 3, 7)
    assert after.get_data_dtype() == np.uint16
    assert np.array_equal(np.asanyarray(after.dataobj), raw[..., 3:])
    assert np.allclose(after.affine, before.affine)
    assert after.header["descrip"] == before.header["descrip"]
    assert after.header["pixdim"][4] == pytest.approx(0.8)


def test_the_sidecar_records_it(tmp_path) -> None:
    image = _bold(tmp_path / "func")
    side = image.with_name("sub-01_task-rest_bold.json")
    data = json.loads(side.read_text())
    data[FIELD] = 2                       # already discarded by someone
    data["VolumeTiming"] = [0.0, 0.8, 1.6, 2.4, 3.2, 4.0, 4.8, 5.6, 6.4, 7.2]
    side.write_text(json.dumps(data))
    drop_leading_volumes([_result(image, 3)])
    data = json.loads(side.read_text())
    assert data[FIELD] == 5
    assert data["VolumeTiming"][:2] == [0.0, 0.8] and len(data["VolumeTiming"]) == 7


def test_the_events_move_with_the_new_first_volume(tmp_path) -> None:
    image = _bold(tmp_path / "func")
    events = image.with_name("sub-01_task-rest_events.tsv")
    events.write_text("onset\tduration\ttrial_type\n1.6\t2\tgo\n10.5\t1\tstop\n")
    drop_leading_volumes([_result(image, 3)])
    rows = [line.split("\t") for line in events.read_text().splitlines()]
    assert rows[0] == ["onset", "duration", "trial_type"]
    # 3 volumes x 0.8 s: an event before the new first volume keeps its
    # negative onset, which the standard allows.
    assert [r[0] for r in rows[1:]] == ["-0.8", "8.1"]
    assert [r[2] for r in rows[1:]] == ["go", "stop"]


def test_a_run_too_short_is_kept_whole(tmp_path, caplog) -> None:
    image = _bold(tmp_path / "func", volumes=3)
    assert drop_leading_volumes([_result(image, 3)]) == 0
    assert nib.load(str(image)).shape[3] == 3
    assert "kept whole" in caplog.text


def test_only_functional_time_series(tmp_path) -> None:
    assert applies("func", "bold") and applies("func", "cbv")
    assert not applies("func", "sbref") and not applies("dwi", "dwi")
    assert not applies("pet", "pet") and not applies("anat", "T1w")
    image = _bold(tmp_path / "dwi")
    assert drop_leading_volumes([_result(image, 3, datatype="dwi", suffix="dwi")]) == 0
    assert nib.load(str(image)).shape[3] == 10


def test_a_cell_wins_over_the_default() -> None:
    row = pd.Series({"drop_volumes": ""})
    assert _drop_volumes_for(row, "func", "bold", 4) == 4
    assert _drop_volumes_for(pd.Series({"drop_volumes": "2"}), "func", "bold", 4) == 2
    assert _drop_volumes_for(pd.Series({"drop_volumes": "0"}), "func", "bold", 4) == 0
    assert _drop_volumes_for(pd.Series({"drop_volumes": "5"}), "anat", "T1w", 4) == 0
    assert _drop_volumes_for(pd.Series({}), "func", "bold", 0) == 0


def test_physiology_follows_its_run(tmp_path) -> None:
    rows = pd.DataFrame([
        {"participant_id": "sub-01", "series_uid": "1.1", "datatype": "func",
         "bids_name": "sub-01_task-rest_bold", "bids_guess_suffix": "bold",
         "drop_volumes": "3", "dataset": "d", "include": "1"},
        {"participant_id": "sub-01", "series_uid": "1.2", "datatype": "func",
         "bids_name": "sub-01_task-rest_physio", "bids_guess_suffix": "physio",
         "drop_volumes": "", "dataset": "d", "include": "1"},
        {"participant_id": "sub-01", "series_uid": "1.3", "datatype": "func",
         "bids_name": "sub-01_task-other_physio", "bids_guess_suffix": "physio",
         "drop_volumes": "", "dataset": "d", "include": "1"},
    ])
    files = {"1.1": [str(tmp_path / "a.dcm")], "1.2": [str(tmp_path / "b.dcm")],
             "1.3": [str(tmp_path / "c.dcm")]}
    tasks = _build_tasks_for_subject(
        rows, tmp_path, files,
        options=TaskOptions(physio_layout="single", physio_align=True))
    by_name = {t.basename: t for t in tasks}
    assert by_name["sub-01_task-rest_bold"].drop_volumes == 3
    assert by_name["sub-01_task-rest_physio"].drop_volumes == 3
    assert by_name["sub-01_task-other_physio"].drop_volumes == 0
    assert by_name["sub-01_task-rest_physio"].physio_layout == "single"
    assert by_name["sub-01_task-rest_physio"].physio_align is True


def test_the_command_line_hands_it_over(monkeypatch, tmp_path) -> None:
    from bidsmgr.cli import convert

    seen = {}
    monkeypatch.setattr(convert, "run_convert",
                        lambda *a, **k: seen.update(k) or 0)
    (tmp_path / "inv.tsv").write_text("participant_id\n")
    convert.main([str(tmp_path / "inv.tsv"), str(tmp_path / "out"), "--drop-volumes", "4",
                  "--physio-layout", "single", "--physio-align", "--no-log-file"])
    assert seen["drop_volumes"] == 4
    assert seen["physio_layout"] == "single" and seen["physio_align"] is True


def test_an_unknown_physio_layout_is_refused(tmp_path) -> None:
    from bidsmgr.cli.convert import run_convert

    (tmp_path / "inv.tsv").write_text("participant_id\tdataset\n")
    with pytest.raises(ValueError, match="unknown physio layout"):
        run_convert(tmp_path / "inv.tsv", tmp_path / "out", physio_layout="per_channel",
                    write_log=False)


def test_the_provenance_records_what_was_asked() -> None:
    import inspect

    from bidsmgr.cli import convert

    source = inspect.getsource(convert._write_provenance)
    for key in ("drop_volumes", "physio_layout", "physio_align"):
        assert f'"{key}"' in source
