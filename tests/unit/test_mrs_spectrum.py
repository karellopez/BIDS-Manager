"""The maths a NIfTI-MRS file goes through before it means anything.

Qt-free, so it runs without a QApplication. The viewer is tested separately;
these are the parts that would be wrong in a way nobody could see.
"""

from __future__ import annotations

import numpy as np
import pytest

from bidsmgr.gui.widgets.mrs_spectrum import (
    DEFAULT_PPM_RANGE_1H,
    METABOLITES_1H,
    apodize,
    combine,
    default_ppm_range,
    is_mrs_path,
    metabolites_for,
    part,
    spectrum,
)

SF_MHZ = 123.26          # a 3 T 1H spectrometer
DWELL = 0.0005           # 2000 Hz spectral width
N = 2048


def _fid_at(ppm_shift: float, *, decay_s: float = 0.15) -> np.ndarray:
    """A decaying complex sinusoid that should transform to one peak at ``ppm``."""
    hz = -(ppm_shift - 4.65) * SF_MHZ
    t = np.arange(N) * DWELL
    return np.exp(2j * np.pi * hz * t) * np.exp(-t / decay_s)


class TestThePpmAxis:
    """The step that is easy to get wrong and impossible to notice."""

    @pytest.mark.parametrize("name,shift", [
        ("NAA", 2.01), ("Cr", 3.03), ("Cho", 3.22), ("mI", 3.56),
    ])
    def test_a_peak_lands_where_it_was_put(self, name, shift):
        ppm, _hz, spec = spectrum(_fid_at(shift), DWELL, SF_MHZ, "1H")
        found = ppm[int(np.argmax(np.abs(spec)))]
        assert found == pytest.approx(shift, abs=0.02), name

    def test_the_axis_is_not_mirrored(self):
        """A spectrum with the sign the other way round is a perfectly
        plausible picture in which every metabolite is on the wrong side of
        the water. Two peaks pin the direction."""
        ppm, _hz, spec = spectrum(
            _fid_at(2.01) + 0.5 * _fid_at(3.22), DWELL, SF_MHZ, "1H",
        )
        mag = np.abs(spec)
        naa = ppm[int(np.argmax(np.where((ppm > 1.8) & (ppm < 2.2), mag, 0)))]
        cho = ppm[int(np.argmax(np.where((ppm > 3.0) & (ppm < 3.4), mag, 0)))]
        assert naa == pytest.approx(2.01, abs=0.02)
        assert cho == pytest.approx(3.22, abs=0.02)
        assert naa < cho          # and NAA really is the lower shift

    def test_water_sits_at_the_reference(self):
        ppm, _hz, spec = spectrum(_fid_at(4.65), DWELL, SF_MHZ, "1H")
        assert ppm[int(np.argmax(np.abs(spec)))] == pytest.approx(4.65, abs=0.02)

    def test_no_spectrometer_frequency_gives_hz_not_nonsense(self):
        """Better an axis in Hz than a ppm scale that means nothing."""
        ppm, hz, _spec = spectrum(_fid_at(2.01), DWELL, 0.0, "1H")
        assert np.allclose(ppm, hz)


class TestProcessing:
    def test_line_broadening_widens_and_shortens_the_peak(self):
        """It trades resolution for signal-to-noise. Both halves checked."""
        fid = _fid_at(2.01)
        _p, _h, plain = spectrum(fid, DWELL, SF_MHZ, "1H")
        _p, _h, broad = spectrum(fid, DWELL, SF_MHZ, "1H", line_broadening_hz=10.0)
        assert np.max(np.abs(broad)) < np.max(np.abs(plain))
        # Wider: more bins above half the maximum.
        def width(s):
            m = np.abs(s)
            return int(np.sum(m > m.max() / 2))
        assert width(broad) > width(plain)

    def test_zero_broadening_changes_nothing(self):
        fid = _fid_at(2.01)
        assert np.array_equal(apodize(fid, DWELL, 0.0), fid)

    def test_phase_rotates_without_changing_magnitude(self):
        fid = _fid_at(2.01)
        _p, _h, a = spectrum(fid, DWELL, SF_MHZ, "1H")
        _p, _h, b = spectrum(fid, DWELL, SF_MHZ, "1H", phase_deg=90.0)
        assert np.allclose(np.abs(a), np.abs(b))
        assert not np.allclose(a.real, b.real)

    def test_the_parts_are_what_they_say(self):
        spec = np.array([1 + 1j, -2 + 0j])
        assert np.allclose(part(spec, "real"), [1, -2])
        assert np.allclose(part(spec, "imaginary"), [1, 0])
        assert np.allclose(part(spec, "magnitude"), [np.sqrt(2), 2])


class TestDynamics:
    def test_the_repeats_average_by_default(self):
        """The repeats exist to be averaged; that is where the SNR is."""
        fid = np.stack([_fid_at(2.01), _fid_at(2.01) * 3], axis=1)
        assert np.allclose(combine(fid), fid.mean(axis=1))

    def test_one_repeat_can_be_singled_out(self):
        """How a corrupted repeat gets found."""
        fid = np.stack([_fid_at(2.01), _fid_at(3.03)], axis=1)
        assert np.allclose(combine(fid, mode="single", index=1), fid[:, 1])

    def test_an_out_of_range_repeat_is_clamped_not_an_error(self):
        fid = np.stack([_fid_at(2.01), _fid_at(3.03)], axis=1)
        assert np.allclose(combine(fid, mode="single", index=99), fid[:, 1])


class TestTheMetaboliteTable:
    def test_the_shifts_are_the_textbook_ones(self):
        table = dict((name, shift) for name, shift, _d in METABOLITES_1H)
        assert table["NAA"] == 2.01
        assert table["Cr"] == 3.03
        assert table["Cho"] == 3.22
        assert table["mI"] == 3.56
        assert table["Lac"] == 1.33

    def test_every_entry_falls_inside_a_readable_window(self):
        for name, shift, _d in METABOLITES_1H:
            assert 0.0 < shift < 5.0, name

    def test_only_1h_gets_labels(self):
        """1H labels on a 31P spectrum would be in the wrong places."""
        assert metabolites_for("1H")
        assert metabolites_for("31P") == ()

    def test_1h_opens_in_the_conventional_window(self):
        ppm = np.linspace(-3.0, 12.0, 100)
        assert default_ppm_range("1H", ppm) == DEFAULT_PPM_RANGE_1H

    def test_another_nucleus_opens_on_its_own_data(self):
        ppm = np.linspace(-20.0, 20.0, 100)
        assert default_ppm_range("31P", ppm) == (-20.0, 20.0)


class TestRouting:
    """Routing a click must never read a volume off disk."""

    def test_an_mrs_file_routes(self, tmp_path):
        p = tmp_path / "sub-01" / "mrs" / "sub-01_svs.nii.gz"
        p.parent.mkdir(parents=True)
        p.write_bytes(b"")
        assert is_mrs_path(p) is True

    def test_a_bold_does_not(self, tmp_path):
        p = tmp_path / "sub-01" / "func" / "sub-01_task-x_bold.nii.gz"
        p.parent.mkdir(parents=True)
        p.write_bytes(b"")
        assert is_mrs_path(p) is False

    def test_an_unknown_suffix_in_an_mrs_folder_does_not(self, tmp_path):
        p = tmp_path / "sub-01" / "mrs" / "sub-01_T1w.nii.gz"
        p.parent.mkdir(parents=True)
        p.write_bytes(b"")
        assert is_mrs_path(p) is False

    def test_the_json_sidecar_does_not(self, tmp_path):
        p = tmp_path / "sub-01" / "mrs" / "sub-01_svs.json"
        p.parent.mkdir(parents=True)
        p.write_bytes(b"")
        assert is_mrs_path(p) is False
