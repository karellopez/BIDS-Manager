"""Tests for the NIfTI viewer's 3-D GPU raycast integration.

These exercise the *wiring* — the 3D toolbar toggle, mutual exclusion
with Multi view, control enable/disable, and the data path into the GL
view — without requiring a live OpenGL context. The ``RaycastGLWidget``
is created (a ``QOpenGLWidget``) but never shown, so ``initializeGL`` /
rendering don't run; the volume is stashed as ``pending`` and observed
via :meth:`RaycastGLWidget.has_volume`. This keeps the suite green under
the headless ``offscreen`` Qt platform used in CI.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

nib = pytest.importorskip("nibabel")
pytest.importorskip("OpenGL")

from bidsmgr.gui.widgets.nifti_viewer_pane import NiftiViewerPane
from bidsmgr.gui.widgets.nifti_gl_view import Nifti3DView, request_gl_format


pytestmark = pytest.mark.gui


def _write_nifti(path: Path, arr: np.ndarray) -> None:
    nib.save(nib.Nifti1Image(arr, affine=np.eye(4)), str(path))


@pytest.fixture
def bids_root_with_nifti(tmp_path: Path) -> Path:
    root = tmp_path / "Studyname"
    anat = root / "sub-01" / "ses-01" / "anat"
    anat.mkdir(parents=True)
    arr_3d = np.arange(10 * 12 * 8, dtype=np.float32).reshape(10, 12, 8)
    _write_nifti(anat / "sub-01_ses-01_T1w.nii.gz", arr_3d)
    func = root / "sub-01" / "ses-01" / "func"
    func.mkdir(parents=True)
    arr_4d = np.random.default_rng(0).random((6, 6, 4, 3), dtype=np.float32)
    _write_nifti(func / "sub-01_ses-01_task-rest_bold.nii.gz", arr_4d)
    return root


def _t1(root: Path) -> Path:
    return root / "sub-01" / "ses-01" / "anat" / "sub-01_ses-01_T1w.nii.gz"


def _bold(root: Path) -> Path:
    return root / "sub-01" / "ses-01" / "func" / "sub-01_ses-01_task-rest_bold.nii.gz"


def _load_and_wait(pane, path, root, qtbot, timeout_ms: int = 5000):
    with qtbot.waitSignal(pane.loaded, timeout=timeout_ms):
        pane.set_file(path, root)


# ---------------------------------------------------------------------------
# request_gl_format
# ---------------------------------------------------------------------------


def test_request_gl_format_is_33_core(qapp) -> None:
    from PyQt6.QtGui import QSurfaceFormat

    fmt = request_gl_format()
    assert fmt.majorVersion() == 3 and fmt.minorVersion() == 3
    assert fmt.profile() == QSurfaceFormat.OpenGLContextProfile.CoreProfile


# ---------------------------------------------------------------------------
# 3D button enablement
# ---------------------------------------------------------------------------


def test_3d_button_disabled_before_load(qapp, isolated_settings) -> None:
    pane = NiftiViewerPane()
    assert pane._td_btn.isEnabled() is False


def test_3d_button_enabled_for_3d(
    qapp, qtbot, bids_root_with_nifti, isolated_settings,
) -> None:
    pane = NiftiViewerPane()
    _load_and_wait(pane, _t1(bids_root_with_nifti), bids_root_with_nifti, qtbot)
    assert pane._td_btn.isEnabled() is True


def test_3d_button_enabled_for_4d(
    qapp, qtbot, bids_root_with_nifti, isolated_settings,
) -> None:
    pane = NiftiViewerPane()
    _load_and_wait(pane, _bold(bids_root_with_nifti), bids_root_with_nifti, qtbot)
    assert pane._td_btn.isEnabled() is True


# ---------------------------------------------------------------------------
# Toggle behaviour
# ---------------------------------------------------------------------------


def test_3d_toggle_switches_stack_and_disables_2d_controls(
    qapp, qtbot, bids_root_with_nifti, isolated_settings,
) -> None:
    pane = NiftiViewerPane()
    _load_and_wait(pane, _t1(bids_root_with_nifti), bids_root_with_nifti, qtbot)

    pane._td_btn.click()
    qapp.processEvents()
    assert pane._three_d is True
    # The GL page was created and is now current.
    assert pane._gl_page_index is not None
    assert pane._image_stack.currentIndex() == pane._gl_page_index
    # 2-D-only controls greyed out.
    for btn in (pane._sa_btn, pane._co_btn, pane._ax_btn):
        assert not btn.isEnabled()
    assert not pane._slice_slider.isEnabled()
    assert not pane._bright_slider.isEnabled()
    assert not pane._contrast_slider.isEnabled()
    # Multi view stays live so the user can switch straight to it.
    assert pane._tri_btn.isEnabled()

    # Toggle off restores the single-pane view + controls.
    pane._td_btn.click()
    qapp.processEvents()
    assert pane._three_d is False
    assert pane._image_stack.currentIndex() == 0
    assert pane._sa_btn.isEnabled()
    assert pane._slice_slider.isEnabled()
    assert pane._bright_slider.isEnabled()


def test_3d_toggle_feeds_volume_to_gl_view(
    qapp, qtbot, bids_root_with_nifti, isolated_settings,
) -> None:
    pane = NiftiViewerPane()
    _load_and_wait(pane, _t1(bids_root_with_nifti), bids_root_with_nifti, qtbot)
    pane._td_btn.click()
    qapp.processEvents()
    assert pane._gl_view is not None
    # Volume was handed to the GL widget (stashed as pending until first paint).
    assert pane._gl_view.gl.has_volume() is True


def test_3d_and_multiview_mutually_exclusive(
    qapp, qtbot, bids_root_with_nifti, isolated_settings,
) -> None:
    pane = NiftiViewerPane()
    _load_and_wait(pane, _t1(bids_root_with_nifti), bids_root_with_nifti, qtbot)

    # Multi view on, then 3D on → Multi view turns off.
    pane._tri_btn.click()
    qapp.processEvents()
    assert pane._tri_view is True
    pane._td_btn.click()
    qapp.processEvents()
    assert pane._three_d is True
    assert pane._tri_view is False
    assert pane._image_stack.currentIndex() == pane._gl_page_index

    # Multi view on again → 3D turns off.
    pane._tri_btn.click()
    qapp.processEvents()
    assert pane._tri_view is True
    assert pane._three_d is False
    assert pane._image_stack.currentIndex() == 1


def test_volume_slider_repushes_in_3d(
    qapp, qtbot, bids_root_with_nifti, isolated_settings,
) -> None:
    """Changing the 4-D volume while 3-D is open re-feeds the GL view."""
    pane = NiftiViewerPane()
    _load_and_wait(pane, _bold(bids_root_with_nifti), bids_root_with_nifti, qtbot)
    pane._td_btn.click()
    qapp.processEvents()
    # Moving the volume slider should not raise and keeps a volume bound.
    pane._vol_slider.setValue(2)
    qapp.processEvents()
    assert pane._gl_view.gl.has_volume() is True


def test_set_file_none_clears_3d(
    qapp, qtbot, bids_root_with_nifti, isolated_settings,
) -> None:
    pane = NiftiViewerPane()
    _load_and_wait(pane, _t1(bids_root_with_nifti), bids_root_with_nifti, qtbot)
    pane._td_btn.click()
    qapp.processEvents()
    assert pane._three_d is True

    pane.set_file(None, None)
    qapp.processEvents()
    assert pane._three_d is False
    assert pane._td_btn.isEnabled() is False
    assert pane._image_stack.currentIndex() == 0


def test_orientation_shortcut_exits_3d(
    qapp, qtbot, bids_root_with_nifti, isolated_settings,
) -> None:
    from bidsmgr.gui.widgets.nifti_viewer_pane import _AXIS_SAGITTAL

    pane = NiftiViewerPane()
    _load_and_wait(pane, _t1(bids_root_with_nifti), bids_root_with_nifti, qtbot)
    pane._td_btn.click()
    qapp.processEvents()
    assert pane._three_d is True
    # Pressing a plane shortcut leaves 3-D and shows that plane in 2-D.
    pane._shortcut_orientation(_AXIS_SAGITTAL)
    qapp.processEvents()
    assert pane._three_d is False
    assert pane._orientation == _AXIS_SAGITTAL
    assert pane._image_stack.currentIndex() == 0


# ---------------------------------------------------------------------------
# Nifti3DView unit
# ---------------------------------------------------------------------------


def test_nifti3dview_normalises_and_binds(qapp) -> None:
    view = Nifti3DView()
    vol = np.linspace(0, 1000, 8 * 8 * 8, dtype=np.float32).reshape(8, 8, 8)
    view.set_volume(vol, (1.0, 1.0, 1.0))
    assert view.gl.has_volume() is True
    # Effect combo drives the GL widget's effect field.
    view.controls._effect.setCurrentIndex(1)   # MIP
    assert view.gl.effect == 1


def test_effect_and_lighting_selectors(qapp) -> None:
    from bidsmgr.gui.widgets.nifti_gl_view import EFFECTS, LIGHTINGS

    view = Nifti3DView()
    assert len(EFFECTS) >= 5 and len(LIGHTINGS) >= 5
    view.controls._effect.setCurrentIndex(2)   # Glass
    assert view.gl.effect == 2
    view.controls._light.setCurrentIndex(3)    # a different matcap
    assert view.gl.matcap_name == LIGHTINGS[3]


def test_threshold_sliders_cannot_invert(qapp) -> None:
    """Regression: lo dragged above hi used to paint the whole box black."""
    view = Nifti3DView()
    c = view.controls
    c._hi.setValue(300)
    c._lo.setValue(900)                        # shove lo past hi
    # The coupling keeps hi strictly above lo, so the shader window is valid.
    assert view.gl.thresh_hi > view.gl.thresh_lo
    assert c._hi.value() > c._lo.value()


def test_clip_controls_drive_gl(qapp) -> None:
    """MRIcroGL-style oblique clip: enable + azimuth/elevation/depth/thick."""
    view = Nifti3DView()
    c = view.controls
    assert view.gl.clip_active == 0
    assert not c._caz.isEnabled()              # az/el/depth/thick off until enabled
    c._clip_en.setChecked(True)
    assert view.gl.clip_active == 1
    assert c._caz.isEnabled() and c._cel.isEnabled()
    # Elevation 90° -> axial normal (0,0,1); az=0 el=0 -> coronal (0,1,0).
    c._cel.setValue(90)
    assert view.gl.clip_normal[2] > 0.99
    c._cel.setValue(0); c._caz.setValue(0)
    assert abs(view.gl.clip_normal[1] - 1.0) < 1e-3
    c._clip_en.setChecked(False)
    assert view.gl.clip_active == 0
    assert not c._caz.isEnabled()


def test_effect_param_graying(qapp) -> None:
    """Per-effect the panel shows only the relevant parameter rows."""
    from bidsmgr.gui.widgets.nifti_gl_view import EFFECTS

    view = Nifti3DView()
    c = view.controls
    c._effect.setCurrentIndex(EFFECTS.index("Glass"))
    assert c._rows["brighten"].isHidden()      # Glass: no matcap brightness
    assert not c._rows["specular"].isHidden()  # Glass: Phong specular shown
    c._effect.setCurrentIndex(EFFECTS.index("MIP"))
    assert c._rows["density"].isHidden()        # MIP: nothing but window + quality
    c._effect.setCurrentIndex(EFFECTS.index("Opacity peeling"))
    assert not c._rows["peel"].isHidden()       # peeling: peel layers shown


def test_slice_overlay_per_effect_default(qapp) -> None:
    """The cut-face intensity slice defaults on for surface effects only."""
    from bidsmgr.gui.widgets.nifti_gl_view import EFFECTS, SLICE_DEFAULT_ON

    view = Nifti3DView()
    c = view.controls
    for name in SLICE_DEFAULT_ON:                 # Default / Matte / Topography
        c._effect.setCurrentIndex(EFFECTS.index(name))
        assert view.gl.slice_overlay == 1
        assert not c._rows["overlaydepth"].isHidden()
    for name in ("Glass", "Shell", "Opacity peeling", "MIP"):
        c._effect.setCurrentIndex(EFFECTS.index(name))
        assert view.gl.slice_overlay == 0
        assert c._rows["overlaydepth"].isHidden()
    c._effect.setCurrentIndex(0)
    c._odepth.setValue(70)                        # overlay-depth slider drives gl
    assert abs(view.gl.slice_depth - 0.70) < 1e-6


def test_gpu_available_and_cube(qapp) -> None:
    from bidsmgr.gui.widgets import nifti_gl_view as M

    # Probe returns a bool and is cached; cube geometry/atlas build.
    assert isinstance(M.gpu_available(), bool)
    assert M._cube_geometry().shape == (36, 8)
    view = Nifti3DView()
    assert view.gl._show_cube is True
    view.set_show_cube(False)
    assert view.gl._show_cube is False


def test_volume_retained_for_context_loss(qapp) -> None:
    """Regression: detach/re-attach recreates the GL context; the widget must
    keep the volume so ``initializeGL`` can re-upload it (else 3-D goes blank).
    """
    w = RaycastGLWidget()
    vol = np.random.default_rng(0).random((8, 8, 8), dtype=np.float32)
    w.set_volume_float(vol, (1.0, 1.0, 1.0))
    assert w._volume is not None            # retained, not consumed on upload
    assert w.has_volume() is True


def test_pan_moves_target(qapp) -> None:
    from PyQt6.QtCore import QPoint, QPointF, Qt
    from PyQt6.QtGui import QMouseEvent

    w = RaycastGLWidget()
    t0 = w._target.copy()
    press = QMouseEvent(
        QMouseEvent.Type.MouseButtonPress, QPointF(0, 0), QPointF(0, 0),
        Qt.MouseButton.RightButton, Qt.MouseButton.RightButton,
        Qt.KeyboardModifier.NoModifier,
    )
    w.mousePressEvent(press)
    assert w._drag == "pan"
    move = QMouseEvent(
        QMouseEvent.Type.MouseMove, QPointF(30, 10), QPointF(30, 10),
        Qt.MouseButton.RightButton, Qt.MouseButton.RightButton,
        Qt.KeyboardModifier.NoModifier,
    )
    w.mouseMoveEvent(move)
    assert not np.allclose(w._target, t0)      # pan shifted the camera target
    w.reset_view()
    assert np.allclose(w._target, 0.0)         # reset recentres it


def test_refresh_and_reset_params(qapp) -> None:
    view = Nifti3DView()
    c = view.controls
    c._effect.setCurrentIndex(2)
    c._bright.setValue(260)
    c._clip_en.setChecked(True)
    assert view.gl.effect == 2 and view.gl.clip_active == 1
    c.reset_params()
    from bidsmgr.gui.widgets.nifti_gl_view import DEFAULTS
    assert view.gl.effect == 0
    assert view.gl.clip_active == 0
    assert c._bright.value() == DEFAULTS["brighten"]
    view.gl.refresh()                          # no-op without a live context, must not raise


def test_drag_right_increases_azimuth(qapp) -> None:
    """Regression: left-right drag was inverted."""
    from PyQt6.QtCore import QPoint, QPointF, Qt
    from PyQt6.QtGui import QMouseEvent

    w = RaycastGLWidget()
    az0 = w._az
    w._last = QPoint(0, 0)
    ev = QMouseEvent(
        QMouseEvent.Type.MouseMove, QPointF(10, 0), QPointF(10, 0),
        Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    w.mouseMoveEvent(ev)
    assert w._az > az0


# ---------------------------------------------------------------------------
# Ortho 3D (3 planes + render) combined mode
# ---------------------------------------------------------------------------


def test_ortho3d_enabled_for_3d(
    qapp, qtbot, bids_root_with_nifti, isolated_settings,
) -> None:
    pane = NiftiViewerPane()
    _load_and_wait(pane, _t1(bids_root_with_nifti), bids_root_with_nifti, qtbot)
    assert pane._quad_btn.isEnabled() is True


def test_ortho3d_toggle_shows_grid_and_keeps_slice_controls(
    qapp, qtbot, bids_root_with_nifti, isolated_settings,
) -> None:
    pane = NiftiViewerPane()
    _load_and_wait(pane, _t1(bids_root_with_nifti), bids_root_with_nifti, qtbot)

    pane._quad_btn.click()
    qapp.processEvents()
    assert pane._combo_view is True
    assert pane._combo_page_index is not None
    assert pane._image_stack.currentIndex() == pane._combo_page_index
    # The render got the volume, and all three slice panels exist.
    assert pane._combo_gl.has_volume() is True
    assert pane._combo_controls is not None
    assert len(pane._combo_labels) == 3
    # Slices are shown, so brightness/contrast/crosshair stay live...
    assert pane._bright_slider.isEnabled()
    assert pane._crosshair_swatch.isEnabled()
    # ...but the single-pane-only controls do not.
    assert not pane._sa_btn.isEnabled()
    assert not pane._slice_slider.isEnabled()


def test_three_3d_modes_mutually_exclusive(
    qapp, qtbot, bids_root_with_nifti, isolated_settings,
) -> None:
    pane = NiftiViewerPane()
    _load_and_wait(pane, _t1(bids_root_with_nifti), bids_root_with_nifti, qtbot)

    pane._quad_btn.click(); qapp.processEvents()
    assert pane._combo_view and not pane._three_d and not pane._tri_view

    pane._td_btn.click(); qapp.processEvents()
    assert pane._three_d and not pane._combo_view

    pane._tri_btn.click(); qapp.processEvents()
    assert pane._tri_view and not pane._three_d and not pane._combo_view

    pane._tri_btn.click(); qapp.processEvents()
    assert not (pane._tri_view or pane._three_d or pane._combo_view)
    assert pane._image_stack.currentIndex() == 0
