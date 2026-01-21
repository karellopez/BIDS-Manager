"""Shared OpenGL helpers for visualization dialogs."""

from dataclasses import dataclass
import importlib.util

from PyQt5.QtWidgets import QCheckBox, QLabel, QSlider


_HAS_PYQTGRAPH = importlib.util.find_spec("pyqtgraph") is not None
if _HAS_PYQTGRAPH:
    import pyqtgraph as pg
    import pyqtgraph.opengl as gl
    from pyqtgraph.opengl import shaders as gl_shaders
else:
    pg = None
    gl = None
    gl_shaders = None

HAS_PYQTGRAPH = _HAS_PYQTGRAPH


def _create_directional_light_shader():
    """Build a lightweight directional lighting shader for ``GLMeshItem``.

    PyQtGraph ships with a fixed light direction inside the built-in
    ``'shaded'`` program which makes the surface difficult to interpret when the
    user wants to change the perceived light source.  Creating our own shader
    allows us to expose the light direction and intensity as uniforms that can
    be updated on the fly without recompiling the OpenGL program each time the
    sliders move.
    """

    if not HAS_PYQTGRAPH or gl is None:
        return None

    shader_mod = getattr(gl, "shaders", None)
    if shader_mod is None:
        return None

    try:
        vertex = shader_mod.VertexShader(
            """
            uniform mat4 u_mvp;
            uniform mat3 u_normal;
            attribute vec4 a_position;
            attribute vec3 a_normal;
            attribute vec4 a_color;
            varying vec3 v_normal;
            varying vec4 v_color;
            void main() {
                v_normal = normalize(u_normal * a_normal);
                v_color = a_color;
                gl_Position = u_mvp * a_position;
            }
            """
        )
        fragment = shader_mod.FragmentShader(
            """
            #ifdef GL_ES
            precision mediump float;
            #endif
            uniform float lightDir[3];
            uniform float lightParams[2];
            varying vec3 v_normal;
            varying vec4 v_color;
            void main() {
                vec3 norm = normalize(v_normal);
                vec3 lightVec = normalize(vec3(lightDir[0], lightDir[1], lightDir[2]));
                float diffuse = max(dot(norm, lightVec), 0.0) * lightParams[0];
                float ambient = lightParams[1];
                float lighting = clamp(ambient + diffuse, 0.0, 1.0);
                vec4 colour = v_color;
                colour.rgb *= lighting;
                gl_FragColor = colour;
            }
            """
        )
    except Exception:  # pragma: no cover - shader compilation errors are runtime only
        return None

    return shader_mod.ShaderProgram(
        None,
        [vertex, fragment],
        uniforms={
            "lightDir": [0.0, 0.0, 1.0],
            # ``lightParams`` stores [diffuse_scale, ambient_strength].  We keep
            # a modest ambient component so that the mesh never becomes entirely
            # black when the light points away from the surface.
            "lightParams": [1.0, 0.35],
        },
    )


def _create_flat_color_shader():
    """Return an unlit shader for meshes when lighting is disabled."""

    if not HAS_PYQTGRAPH or gl is None:
        return None

    shader_mod = getattr(gl, "shaders", None)
    if shader_mod is None:
        return None

    try:
        vertex = shader_mod.VertexShader(
            """
            uniform mat4 u_mvp;
            attribute vec4 a_position;
            attribute vec4 a_color;
            varying vec4 v_color;
            void main() {
                v_color = a_color;
                gl_Position = u_mvp * a_position;
            }
            """
        )
        fragment = shader_mod.FragmentShader(
            """
            #ifdef GL_ES
            precision mediump float;
            #endif
            varying vec4 v_color;
            void main() {
                gl_FragColor = v_color;
            }
            """
        )
    except Exception:  # pragma: no cover - shader compilation errors occur at runtime
        return None

    return shader_mod.ShaderProgram(None, [vertex, fragment], uniforms={})


_SLICE_ORIENTATIONS = (
    ("sagittal", 0, "Left", "Right"),
    ("coronal", 1, "Posterior", "Anterior"),
    ("axial", 2, "Inferior", "Superior"),
)


if HAS_PYQTGRAPH:

    class _AdjustableAxisItem(gl.GLAxisItem):
        """Axis item with configurable line width for better visibility."""

        def __init__(self, *args, **kwargs):
            self._line_width = 2.0
            super().__init__(*args, **kwargs)

        def setLineWidth(self, width: float) -> None:
            self._line_width = max(1.0, float(width))
            self.update()

        def paint(self):  # pragma: no cover - requires OpenGL context
            from OpenGL.GL import (
                GL_LINES,
                GL_LINE_SMOOTH,
                GL_LINE_SMOOTH_HINT,
                GL_NICEST,
                glBegin,
                glColor4f,
                glEnable,
                glEnd,
                glHint,
                glLineWidth,
                glVertex3f,
            )

            self.setupGLState()
            if self.antialias:
                glEnable(GL_LINE_SMOOTH)
                glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
            glLineWidth(self._line_width)
            glBegin(GL_LINES)
            x, y, z = self.size()
            glColor4f(0, 1, 0, 0.6)
            glVertex3f(0, 0, 0)
            glVertex3f(0, 0, z)
            glColor4f(1, 1, 0, 0.6)
            glVertex3f(0, 0, 0)
            glVertex3f(0, y, 0)
            glColor4f(0, 0, 1, 0.6)
            glVertex3f(0, 0, 0)
            glVertex3f(x, 0, 0)
            glEnd()
else:
    _AdjustableAxisItem = None


@dataclass
class _SliceControl:
    """Stores widgets controlling a single anatomical slicer."""

    checkbox: QCheckBox
    min_slider: QSlider
    max_slider: QSlider
    min_value: QLabel
    max_value: QLabel
    axis: int
    negative_name: str
    positive_name: str
