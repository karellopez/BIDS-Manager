"""Visualization subpackage for BIDS Manager."""

from .freesurfer_surface import FreeSurferSurfaceDialog
from .nifti_viewer import NiftiViewer
from .surface_3d import Surface3DDialog
from .volume_3d import Volume3DDialog
from .widgets import ShrinkableScrollArea

__all__ = [
    "FreeSurferSurfaceDialog",
    "NiftiViewer",
    "Surface3DDialog",
    "Volume3DDialog",
    "ShrinkableScrollArea",
]
