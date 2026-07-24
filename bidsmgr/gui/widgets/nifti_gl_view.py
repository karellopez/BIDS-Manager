"""GPU volume-raycasting 3-D view for the NIfTI viewer.

Companion to :class:`bidsmgr.gui.widgets.NiftiViewerPane`. Renders the
current volume with a single-pass OpenGL ray-caster — the technique
MRIcroGL uses — with only the dependencies BIDS-Manager already ships
(PyQt6 + PyOpenGL). No VTK, no WebGL, no new wheels.

Rendering effects (``uEffect``), à la MRIcroGL's shader menu:
    * **Surface** — matcap-lit iso-surface compositing (``Default.glsl``).
    * **MIP**     — maximum-intensity projection (``MIP.glsl``).
    * **Glass**   — translucent surface with a Fresnel rim (``Glass.glsl``).
    * **X-ray**   — integrated attenuation (additive, see-through).
    * **Edges**   — gradient-magnitude silhouettes (``Edges.glsl``).

Lighting is a **matcap** (material-capture) texture sampled by the
view-space gradient normal — MRIcroGL's approach. Several studio materials
are generated procedurally (Shiny White, Clay, Matte, Titanium, Gold, Blue).

**Clip plane** (MRIcroGL ``applyClip``): an oblique plane set by Azimuth /
Elevation / Depth / Thickness. Where the plane cuts through tissue the
exposed face is drawn as a smooth *intensity slice* (like a 2-D cross-
section) rather than a noisy interior iso-surface — the key to MRIcroGL's
clean cut look.

Robustness: the GL context is created lazily and every GL entry point is
guarded, so a missing driver / headless ``offscreen`` platform degrades to
an inert widget instead of crashing. The last volume is retained so the
render survives context loss (window detach / re-attach → ``initializeGL``
re-runs and re-uploads it).

Interaction: left-drag orbit · right/middle-drag pan · wheel zoom.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from PyQt6.QtCore import Qt, QPoint
from PyQt6.QtGui import QSurfaceFormat
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

log = logging.getLogger(__name__)


def request_gl_format() -> QSurfaceFormat:
    """Return (and register as default) an OpenGL 3.3 core surface format.

    macOS only exposes the modern ``#version 330`` shading language through
    a *core-profile* context; the compatibility profile is stuck at 2.1.
    Register this before the ``QApplication`` is built. Safe to call twice.
    """
    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    fmt.setDepthBufferSize(24)
    QSurfaceFormat.setDefaultFormat(fmt)
    return fmt


# Effects (index == uEffect in the shader).
EFFECTS = ["Surface", "MIP", "Glass", "X-ray", "Edges"]

# Matcap "lighting" materials: (ambient, key, fill, spec_power, spec_int, tint).
_LIGHTINGS: dict[str, tuple] = {
    "Shiny White": (0.34, 0.78, 0.26, 90.0, 1.05, (1.00, 0.99, 0.97)),
    "Clay":        (0.38, 0.64, 0.28, 12.0, 0.16, (0.86, 0.78, 0.70)),
    "Matte":       (0.48, 0.54, 0.32,  4.0, 0.00, (0.82, 0.82, 0.84)),
    "Titanium":    (0.24, 0.82, 0.22, 96.0, 0.95, (0.72, 0.75, 0.80)),
    "Gold":        (0.30, 0.74, 0.22, 44.0, 0.85, (0.96, 0.80, 0.36)),
    "Blue":        (0.30, 0.68, 0.28, 34.0, 0.60, (0.56, 0.69, 0.96)),
}
LIGHTINGS = list(_LIGHTINGS)

# Default control values (used by "Reset parameters").
DEFAULTS = dict(
    effect=0, light=0, lo=120, hi=420, den=100, bright=140, surf=65, q=512,
    clip=False, az=0, el=0, depth=500, thick=1000,
)


# --------------------------------------------------------------------------
# Volume normalisation + matrix helpers
# --------------------------------------------------------------------------

def normalize_to_u8(vol: np.ndarray) -> np.ndarray:
    """Window a float volume to uint8 on a robust 0.5–99.5 percentile range."""
    v = np.ascontiguousarray(vol, dtype=np.float32)
    lo, hi = np.percentile(v, (0.5, 99.5))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
        if hi <= lo:
            hi = lo + 1.0
    return (np.clip((v - lo) / (hi - lo), 0.0, 1.0) * 255.0).astype(np.uint8)


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def clip_normal_from(az_deg: float, el_deg: float) -> tuple[float, float, float]:
    """Unit clip-plane normal from azimuth/elevation (RAS texcoord space)."""
    az, el = np.radians(az_deg), np.radians(el_deg)
    ce = np.cos(el)
    n = np.array([ce * np.sin(az), ce * np.cos(az), np.sin(el)], np.float32)
    n = _normalize(n)
    return float(n[0]), float(n[1]), float(n[2])


def _perspective(fovy_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / np.tan(np.radians(fovy_deg) / 2.0)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / max(aspect, 1e-6)
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2.0 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


def _look_at(eye: np.ndarray, center: np.ndarray, up: np.ndarray) -> np.ndarray:
    f = _normalize(center - eye)
    s = _normalize(np.cross(f, up))
    u = np.cross(s, f)
    m = np.eye(4, dtype=np.float32)
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = -f
    m[0, 3] = -np.dot(s, eye)
    m[1, 3] = -np.dot(u, eye)
    m[2, 3] = np.dot(f, eye)
    return m


def _make_matcap(name: str, size: int = 256) -> np.ndarray:
    """Procedural studio-lit sphere → an RGB matcap (``size×size×3`` uint8).

    A matcap encodes the shaded appearance of a unit sphere: for a screen-
    space normal ``(nx, ny)`` the lit colour is read straight out of this
    image. Two directional lights + ambient + a Blinn-Phong highlight give
    MRIcroGL's soft "clay / shiny / metal" looks — parameterised per material
    in :data:`_LIGHTINGS`. A broad soft key plus a tight bright hotspot give
    "Shiny White" the glossy sheen MRIcroGL's default has.
    """
    amb, key_i, fill_i, spow, spec_i, tint = _LIGHTINGS[name]
    ax = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    u, v = np.meshgrid(ax, ax)
    r2 = u * u + v * v
    inside = r2 <= 1.0
    z = np.sqrt(np.clip(1.0 - r2, 0.0, 1.0))
    n = np.stack([u, v, z], axis=-1)

    key = _normalize(np.array([0.35, 0.55, 0.75], np.float32))
    fill = _normalize(np.array([-0.6, 0.10, 0.55], np.float32))
    view = np.array([0.0, 0.0, 1.0], np.float32)
    half = _normalize(key + view)

    ndl_key = np.clip((n * key).sum(-1), 0.0, 1.0)
    ndl_fill = np.clip((n * fill).sum(-1), 0.0, 1.0)
    nh = np.clip((n * half).sum(-1), 0.0, 1.0)
    spec = nh ** spow + 0.25 * (nh ** (spow * 0.25))   # tight hotspot + soft sheen

    # Wrap-diffuse (soft terminator) reads more like MRIcroGL than hard N·L.
    wrap = np.clip((ndl_key + 0.35) / 1.35, 0.0, 1.0)
    lum = amb + key_i * wrap + fill_i * ndl_fill
    col = lum[..., None] * np.array(tint, np.float32)
    col = col + spec[..., None] * (spec_i * np.array([1.0, 1.0, 1.0], np.float32))
    col = np.clip(col, 0.0, 1.0)
    col[~inside] = 0.0
    return (col * 255.0).astype(np.uint8)


# --------------------------------------------------------------------------
# Shaders
# --------------------------------------------------------------------------

_VERT = """
#version 330 core
out vec2 vNdc;
void main() {
    vec2 pos = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    vNdc = pos * 2.0 - 1.0;
    gl_Position = vec4(vNdc, 0.0, 1.0);
}
"""

_FRAG = """
#version 330 core
in  vec2 vNdc;
out vec4 FragColor;

uniform sampler3D uVol;
uniform sampler2D uMatcap;
uniform mat4  uInvViewProj;
uniform mat3  uNormalMatrix;
uniform vec3  uBoxHalf;
uniform vec3  uTexSize;
uniform int   uEffect;         // 0 Surface, 1 MIP, 2 Glass, 3 X-ray, 4 Edges
uniform float uThreshLo;
uniform float uThreshHi;
uniform float uDensity;
uniform float uBrighten;
uniform float uSurface;
uniform int   uSteps;
uniform vec3  uBg;
uniform int   uClipActive;
uniform vec3  uClipNormal;
uniform float uClipDepth;      // signed distance of the near cut plane
uniform float uClipThick;      // slab thickness removed

bool intersectBox(vec3 ro, vec3 rd, out float tNear, out float tFar) {
    vec3 inv = 1.0 / rd;
    vec3 t0 = (-uBoxHalf - ro) * inv;
    vec3 t1 = ( uBoxHalf - ro) * inv;
    vec3 tsm = min(t0, t1), tbg = max(t0, t1);
    tNear = max(max(tsm.x, tsm.y), tsm.z);
    tFar  = min(min(tbg.x, tbg.y), tbg.z);
    return tFar >= max(tNear, 0.0);
}

float samp(vec3 uvw) { return texture(uVol, uvw).r; }

bool clipped(vec3 uvw) {
    if (uClipActive == 0) return false;
    float sd = dot(uClipNormal, uvw - vec3(0.5));
    return sd > uClipDepth && sd < uClipDepth + uClipThick;
}

vec3 gradient(vec3 uvw) {
    vec3 e = 1.0 / uTexSize;
    return vec3(
        samp(uvw + vec3(e.x, 0, 0)) - samp(uvw - vec3(e.x, 0, 0)),
        samp(uvw + vec3(0, e.y, 0)) - samp(uvw - vec3(0, e.y, 0)),
        samp(uvw + vec3(0, 0, e.z)) - samp(uvw - vec3(0, 0, e.z)));
}

float hash(vec2 p) {
    return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
}

void main() {
    vec4 pn = uInvViewProj * vec4(vNdc, -1.0, 1.0);
    vec4 pf = uInvViewProj * vec4(vNdc,  1.0, 1.0);
    vec3 ro = pn.xyz / pn.w;
    vec3 rd = normalize(pf.xyz / pf.w - ro);

    float tNear, tFar;
    if (!intersectBox(ro, rd, tNear, tFar)) { FragColor = vec4(uBg, 1.0); return; }
    tNear = max(tNear, 0.0);

    vec3  boxSize = 2.0 * uBoxHalf;
    float dt      = length(boxSize) / float(uSteps);
    float refStep = length(boxSize) / 512.0;
    float t0      = tNear + hash(gl_FragCoord.xy) * dt;

    // Guard the transfer-function window so a lo>=hi drag can't invert the
    // smoothstep and paint the whole box (the "black square" bug).
    float e0 = min(uThreshLo, uThreshHi);
    float e1 = max(uThreshLo, uThreshHi);
    if (e1 <= e0) e1 = e0 + 1e-3;

    // ---- MIP ----
    if (uEffect == 1) {
        float mx = 0.0, t = t0;
        for (int i = 0; i < 4096; ++i) {
            if (t > tFar) break;
            vec3 uvw = (ro + rd * t + uBoxHalf) / boxSize;
            if (!clipped(uvw)) mx = max(mx, samp(uvw));
            t += dt;
        }
        float w = clamp((mx - e0) / (e1 - e0), 0.0, 1.0);
        FragColor = vec4(mix(uBg, vec3(1.0), w), 1.0);
        return;
    }

    // ---- X-ray ----
    if (uEffect == 3) {
        float sum = 0.0, t = t0;
        for (int i = 0; i < 4096; ++i) {
            if (t > tFar) break;
            vec3 uvw = (ro + rd * t + uBoxHalf) / boxSize;
            if (!clipped(uvw)) { float d = samp(uvw); if (d > e0) sum += (d - e0) * uDensity; }
            t += dt;
        }
        float a = 1.0 - exp(-sum * dt * 6.0);
        FragColor = vec4(mix(uBg, vec3(1.0), clamp(a, 0.0, 1.0)), 1.0);
        return;
    }

    // ---- Surface (0) / Glass (2) / Edges (4): compositing ----
    bool glass = (uEffect == 2);
    bool edges = (uEffect == 4);
    float aScale = glass ? 0.4 : 1.0;
    vec4 acc = vec4(0.0);
    float t = t0;
    bool prevClip = false;
    for (int i = 0; i < 4096; ++i) {
        if (t > tFar || (acc.a > 0.985 && !glass)) break;
        vec3 uvw = (ro + rd * t + uBoxHalf) / boxSize;
        if (clipped(uvw)) { prevClip = true; t += dt; continue; }
        float d = samp(uvw);

        // Cut face: first kept sample right after crossing the clip plane.
        // Show a smooth intensity cross-section (MRIcroGL slice look) where
        // tissue is present; fall through to the 3-D surface where it's air.
        if (uClipActive == 1 && prevClip && !edges) {
            prevClip = false;
            if (d > 0.05) {
                float sv = pow(clamp(d, 0.0, 1.0), 0.8);
                acc.rgb += (1.0 - acc.a) * vec3(sv);
                acc.a = 1.0;
                break;
            }
        }
        prevClip = false;

        float a = smoothstep(e0, e1, d) * uDensity * aScale;
        if (a > 0.001) {
            a = 1.0 - pow(1.0 - a, dt / refStep);
            vec3 nw = normalize(-gradient(uvw) + 1e-6);
            vec3 nv = normalize(uNormalMatrix * nw);
            float fres = pow(1.0 - abs(nv.z), 2.5);
            vec3 lit;
            if (edges) {
                lit = vec3(fres) * uBrighten;
            } else {
                vec3 mc = texture(uMatcap, nv.xy * 0.5 + 0.5).rgb;
                vec3 surf = mix(vec3(0.5), vec3(pow(d, 0.8)), uSurface);
                lit = mc * surf * uBrighten;
                if (glass) lit += vec3(0.5) * fres;
                // Depth cue: dim samples deeper into the volume for shape.
                float depth01 = clamp((t - tNear) / max(tFar - tNear, 1e-3), 0.0, 1.0);
                lit *= 1.0 - 0.3 * depth01;
            }
            acc.rgb += (1.0 - acc.a) * lit * a;
            acc.a   += (1.0 - acc.a) * a;
        }
        t += dt;
    }
    FragColor = vec4(acc.rgb + (1.0 - acc.a) * uBg, 1.0);
}
"""


# --------------------------------------------------------------------------
# GL widget (bare canvas)
# --------------------------------------------------------------------------

class RaycastGLWidget(QOpenGLWidget):
    """QOpenGLWidget hosting the single-pass volume raycaster (no controls)."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFormat(request_gl_format())

        self._prog = 0
        self._vao = 0
        self._tex = 0
        self._matcap_tex = 0
        self._tex_dims = (1, 1, 1)
        self._box_half = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        self._gl_ok = False
        self._gl_error = ""
        # Retained so the render survives context loss (detach / re-attach):
        # ``initializeGL`` re-runs on the new context and re-uploads this.
        self._volume: Optional[tuple[np.ndarray, tuple[float, float, float]]] = None
        self._matcaps: dict[str, np.ndarray] = {}
        self._matcap_dirty = True

        # Camera: orbit (azimuth/elevation/distance) about a pannable target.
        self._az = 0.6
        self._el = 0.3
        self._dist = 2.6
        self._target = np.zeros(3, np.float32)
        self._last: Optional[QPoint] = None
        self._drag = "rotate"

        # Shader-driven display state (mutated by Nifti3DControls).
        self.effect = DEFAULTS["effect"]
        self.thresh_lo = DEFAULTS["lo"] / 1000.0
        self.thresh_hi = DEFAULTS["hi"] / 1000.0
        self.density = DEFAULTS["den"] / 100.0
        self.brighten = DEFAULTS["bright"] / 100.0
        self.surface = DEFAULTS["surf"] / 100.0
        self.steps = DEFAULTS["q"]
        self.matcap_name = LIGHTINGS[0]
        self.clip_active = 0
        self.clip_normal = (0.0, 1.0, 0.0)
        self.clip_depth = 0.0
        self.clip_thick = 3.0
        self._bg = (0.05, 0.06, 0.08)

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    # -- public API --------------------------------------------------------

    def gl_ok(self) -> bool:
        return self._gl_ok

    def has_volume(self) -> bool:
        return self._volume is not None

    def set_volume(self, u8: np.ndarray, spacing: tuple[float, float, float]) -> None:
        """Bind a uint8 volume; retained + uploaded when GL is ready."""
        self._volume = (np.ascontiguousarray(u8, dtype=np.uint8), spacing)
        if self._prog and self._gl_ok and self._make_current():
            try:
                self._upload_volume()
            finally:
                self.doneCurrent()
            self.update()

    def set_volume_float(self, vol: np.ndarray,
                         spacing: tuple[float, float, float]) -> None:
        self.set_volume(normalize_to_u8(vol), spacing)

    def set_matcap(self, name: str) -> None:
        if name not in _LIGHTINGS:
            return
        self.matcap_name = name
        self._matcap_dirty = True
        if self._prog and self._gl_ok and self._make_current():
            try:
                self._refresh_matcap()
            finally:
                self.doneCurrent()
            self.update()

    def refresh(self) -> None:
        """Re-upload the matcap + volume and repaint (manual recovery)."""
        self._matcap_dirty = True
        if self._prog and self._gl_ok and self._make_current():
            try:
                self._refresh_matcap()
                if self._volume is not None:
                    self._upload_volume()
            finally:
                self.doneCurrent()
        self.update()

    def clear(self) -> None:
        self._volume = None
        if self._tex and self._make_current():
            try:
                self._safe(lambda: GL.glDeleteTextures([self._tex]))
                self._tex = 0
            finally:
                self.doneCurrent()
            self.update()

    def reset_view(self) -> None:
        self._az, self._el, self._dist = 0.6, 0.3, 2.6
        self._target = np.zeros(3, np.float32)
        self.update()

    # -- GL lifecycle ------------------------------------------------------

    def initializeGL(self) -> None:
        # A fresh context (first show, or after a detach/re-attach) invalidates
        # every prior GL object — rebuild them all and re-upload the volume.
        self._gl_ok = False
        self._tex = 0
        try:
            self._import_gl()
            self._prog = self._build_program(_VERT, _FRAG)
            self._vao = GL.glGenVertexArrays(1)
            self._matcap_tex = GL.glGenTextures(1)
            self._matcap_dirty = True
            self._refresh_matcap()
            GL.glDisable(GL.GL_DEPTH_TEST)
            GL.glClearColor(*self._bg, 1.0)
            self._gl_ok = True
            if self._volume is not None:
                self._upload_volume()
        except Exception as exc:  # noqa: BLE001 - degrade, never crash the app
            self._gl_ok = False
            self._gl_error = str(exc)
            log.warning("3-D GL initialisation failed: %s", exc)

    def resizeGL(self, w: int, h: int) -> None:
        if self._gl_ok:
            GL.glViewport(0, 0, w, h)

    def paintGL(self) -> None:
        if not self._gl_ok:
            return
        GL.glClearColor(*self._bg, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        if not self._tex:
            return

        w = max(self.width(), 1)
        h = max(self.height(), 1)
        eye = self._eye()
        view = _look_at(eye, self._target, np.array([0, 0, 1], np.float32))
        proj = _perspective(45.0, w / h, 0.01, 20.0)
        inv_vp = np.linalg.inv(proj @ view).astype(np.float32)
        normal_mat = np.ascontiguousarray(view[:3, :3], np.float32)

        GL.glUseProgram(self._prog)
        self._set_mat4("uInvViewProj", inv_vp)
        self._set_mat3("uNormalMatrix", normal_mat)
        self._set_vec3("uBoxHalf", self._box_half)
        self._set_vec3("uTexSize", np.array(self._tex_dims, np.float32))
        self._set_vec3("uBg", np.array(self._bg, np.float32))
        GL.glUniform1i(self._loc("uEffect"), int(self.effect))
        GL.glUniform1f(self._loc("uThreshLo"), float(self.thresh_lo))
        GL.glUniform1f(self._loc("uThreshHi"), float(self.thresh_hi))
        GL.glUniform1f(self._loc("uDensity"), float(self.density))
        GL.glUniform1f(self._loc("uBrighten"), float(self.brighten))
        GL.glUniform1f(self._loc("uSurface"), float(self.surface))
        GL.glUniform1i(self._loc("uSteps"), int(self.steps))
        GL.glUniform1i(self._loc("uClipActive"), int(self.clip_active))
        self._set_vec3("uClipNormal", np.array(self.clip_normal, np.float32))
        GL.glUniform1f(self._loc("uClipDepth"), float(self.clip_depth))
        GL.glUniform1f(self._loc("uClipThick"), float(self.clip_thick))

        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_3D, self._tex)
        GL.glUniform1i(self._loc("uVol"), 0)
        GL.glActiveTexture(GL.GL_TEXTURE1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._matcap_tex)
        GL.glUniform1i(self._loc("uMatcap"), 1)

        GL.glBindVertexArray(self._vao)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)
        GL.glBindVertexArray(0)

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _import_gl() -> None:
        global GL
        from OpenGL import GL as _GL
        GL = _GL

    def _make_current(self) -> bool:
        try:
            self.makeCurrent()
            return self.context() is not None and self.context().isValid()
        except Exception:  # noqa: BLE001
            return False

    def _safe(self, fn):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            log.debug("GL call failed: %s", exc)
            return None

    def _eye(self) -> np.ndarray:
        ce = np.cos(self._el)
        d = np.array([ce * np.sin(self._az), ce * np.cos(self._az), np.sin(self._el)])
        return (self._target + self._dist * d).astype(np.float32)

    def _camera_basis(self) -> tuple[np.ndarray, np.ndarray]:
        eye = self._eye()
        up = np.array([0, 0, 1], np.float32)
        f = _normalize(self._target - eye)
        r = _normalize(np.cross(f, up))
        u = np.cross(r, f)
        return r, u

    def _refresh_matcap(self) -> None:
        if not self._matcap_dirty or not self._matcap_tex:
            return
        mc = self._matcaps.get(self.matcap_name)
        if mc is None:
            mc = _make_matcap(self.matcap_name, 256)
            self._matcaps[self.matcap_name] = mc
        h, w = mc.shape[:2]
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._matcap_tex)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB8, w, h, 0,
                        GL.GL_RGB, GL.GL_UNSIGNED_BYTE, np.ascontiguousarray(mc))
        self._matcap_dirty = False

    def _upload_volume(self) -> None:
        if self._volume is None:
            return
        u8, spacing = self._volume
        x, y, z = u8.shape
        self._tex_dims = (x, y, z)
        extent = np.array([x * spacing[0], y * spacing[1], z * spacing[2]], np.float32)
        self._box_half = (0.5 * extent / float(extent.max())).astype(np.float32)
        data = np.ascontiguousarray(u8.transpose(2, 1, 0))  # x fastest for GL
        if self._tex:
            GL.glDeleteTextures([self._tex])
        self._tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_3D, self._tex)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        for pname in (GL.GL_TEXTURE_WRAP_S, GL.GL_TEXTURE_WRAP_T, GL.GL_TEXTURE_WRAP_R):
            GL.glTexParameteri(GL.GL_TEXTURE_3D, pname, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexImage3D(GL.GL_TEXTURE_3D, 0, GL.GL_R8, x, y, z, 0,
                        GL.GL_RED, GL.GL_UNSIGNED_BYTE, data)

    def _build_program(self, vsrc: str, fsrc: str) -> int:
        def compile_stage(src: str, stage: int) -> int:
            sh = GL.glCreateShader(stage)
            GL.glShaderSource(sh, src)
            GL.glCompileShader(sh)
            if not GL.glGetShaderiv(sh, GL.GL_COMPILE_STATUS):
                raise RuntimeError(GL.glGetShaderInfoLog(sh).decode())
            return sh

        vs = compile_stage(vsrc, GL.GL_VERTEX_SHADER)
        fs = compile_stage(fsrc, GL.GL_FRAGMENT_SHADER)
        prog = GL.glCreateProgram()
        GL.glAttachShader(prog, vs)
        GL.glAttachShader(prog, fs)
        GL.glLinkProgram(prog)
        if not GL.glGetProgramiv(prog, GL.GL_LINK_STATUS):
            raise RuntimeError(GL.glGetProgramInfoLog(prog).decode())
        GL.glDeleteShader(vs)
        GL.glDeleteShader(fs)
        return prog

    def _loc(self, name: str) -> int:
        return GL.glGetUniformLocation(self._prog, name)

    def _set_mat4(self, name: str, m: np.ndarray) -> None:
        GL.glUniformMatrix4fv(self._loc(name), 1, GL.GL_TRUE,
                              np.ascontiguousarray(m, np.float32))

    def _set_mat3(self, name: str, m: np.ndarray) -> None:
        GL.glUniformMatrix3fv(self._loc(name), 1, GL.GL_TRUE,
                              np.ascontiguousarray(m, np.float32))

    def _set_vec3(self, name: str, v: np.ndarray) -> None:
        GL.glUniform3f(self._loc(name), float(v[0]), float(v[1]), float(v[2]))

    # -- interaction -------------------------------------------------------

    def mousePressEvent(self, ev) -> None:
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        self._last = ev.position().toPoint()
        btn = ev.button()
        pan = (btn in (Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton)
               or ev.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        self._drag = "pan" if pan else "rotate"

    def mouseMoveEvent(self, ev) -> None:
        if self._last is None:
            return
        p = ev.position().toPoint()
        dx = p.x() - self._last.x()
        dy = p.y() - self._last.y()
        self._last = p
        if self._drag == "pan":
            r, u = self._camera_basis()
            scale = 0.0022 * self._dist
            self._target += (-dx * r + dy * u) * scale
        else:
            # Drag right -> volume turns right (camera orbits the other way).
            self._az += dx * 0.01
            self._el = float(np.clip(self._el + dy * 0.01, -1.55, 1.55))
        self.update()

    def mouseReleaseEvent(self, ev) -> None:
        self._last = None

    def wheelEvent(self, ev) -> None:
        factor = 0.9 if ev.angleDelta().y() > 0 else 1.1
        self._dist = float(np.clip(self._dist * factor, 0.5, 12.0))
        self.update()


# --------------------------------------------------------------------------
# Controls (bound to a RaycastGLWidget; laid out as a vertical column)
# --------------------------------------------------------------------------

_THRESH_GAP = 20  # min separation (slider units) between lo and hi


class Nifti3DControls(QWidget):
    """Effect / lighting / transfer-function / clip controls for a render.

    Bound to a :class:`RaycastGLWidget` and stacked vertically so the growing
    control set (now including the MRIcroGL-style clip Azimuth / Elevation /
    Depth / Thickness) stays legible; used inside a scroll area by the pane.
    ``vertical`` is retained for API symmetry (layout is always a column).
    """

    def __init__(self, gl: RaycastGLWidget, *, vertical: bool = True,
                 parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("sidecar-toolbar")
        self.gl = gl
        self._vertical = vertical
        self._build_widgets()
        self._layout()
        self._apply_clip()

    # -- widget construction ----------------------------------------------

    def _build_widgets(self) -> None:
        self._effect = QComboBox()
        self._effect.addItems(EFFECTS)
        self._effect.setToolTip("Rendering effect (Surface, MIP, Glass, X-ray, Edges).")
        self._effect.currentIndexChanged.connect(self._on_effect)

        self._light = QComboBox()
        self._light.addItems(LIGHTINGS)
        self._light.setToolTip("Lighting material (matcap).")
        self._light.currentIndexChanged.connect(self._on_light)

        self._lo = self._slider(0, 1000, DEFAULTS["lo"], self._on_lo)
        self._hi = self._slider(0, 1000, DEFAULTS["hi"], self._on_hi)
        self._den = self._slider(0, 300, DEFAULTS["den"], self._on_den)
        self._bright = self._slider(50, 300, DEFAULTS["bright"], self._on_bright)
        self._surf = self._slider(0, 100, DEFAULTS["surf"], self._on_surf)
        self._q = self._slider(64, 1024, DEFAULTS["q"], self._on_q)

        # Clip: MRIcroGL-style oblique plane (enable + az / el / depth / thick).
        self._clip_enable = QCheckBox("Enable clip plane")
        self._clip_enable.setToolTip("Slice into the volume to reveal interior.")
        self._clip_enable.stateChanged.connect(self._on_clip_change)
        self._az = self._slider(0, 360, DEFAULTS["az"], self._on_clip_change)
        self._el = self._slider(-90, 90, DEFAULTS["el"], self._on_clip_change)
        self._depth = self._slider(0, 1000, DEFAULTS["depth"], self._on_clip_change)
        self._thick = self._slider(0, 1000, DEFAULTS["thick"], self._on_clip_change)

        self._refresh_btn = QPushButton("Refresh image")
        self._refresh_btn.setObjectName("tb-btn-toggle")
        self._refresh_btn.setToolTip("Re-upload + repaint the render.")
        self._refresh_btn.clicked.connect(lambda: self.gl.refresh())

        self._reset_params_btn = QPushButton("Reset parameters")
        self._reset_params_btn.setObjectName("tb-btn-toggle")
        self._reset_params_btn.setToolTip("Restore all render settings to defaults.")
        self._reset_params_btn.clicked.connect(self.reset_params)

        self._reset_view_btn = QPushButton("Reset view")
        self._reset_view_btn.setObjectName("tb-btn-toggle")
        self._reset_view_btn.setToolTip("Recentre the camera.")
        self._reset_view_btn.clicked.connect(self.gl.reset_view)

    def _layout(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 10, 12, 10)
        outer.setSpacing(9)
        for lbl, wdg in (
            ("Effect", self._effect), ("Lighting", self._light),
            ("Threshold low", self._lo), ("Threshold high", self._hi),
            ("Density", self._den), ("Brightness", self._bright),
            ("Surface colour", self._surf), ("Quality", self._q),
        ):
            outer.addLayout(self._stacked(lbl, wdg))

        outer.addSpacing(4)
        outer.addWidget(self._clip_enable)
        for lbl, wdg in (
            ("Clip azimuth", self._az), ("Clip elevation", self._el),
            ("Clip depth", self._depth), ("Clip thickness", self._thick),
        ):
            outer.addLayout(self._stacked(lbl, wdg))

        outer.addSpacing(6)
        outer.addWidget(self._refresh_btn)
        outer.addWidget(self._reset_params_btn)
        outer.addWidget(self._reset_view_btn)
        outer.addStretch(1)
        hint = self._chip("Left-drag rotate · right-drag pan · scroll zoom")
        hint.setWordWrap(True)
        outer.addWidget(hint)

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _chip(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName("sidecar-footer-summary")
        return lbl

    def _stacked(self, text: str, wdg: QWidget) -> QVBoxLayout:
        box = QVBoxLayout()
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(2)
        box.addWidget(self._chip(text))
        box.addWidget(wdg)
        return box

    @staticmethod
    def _slider(lo: int, hi: int, val: int, cb) -> QSlider:
        s = QSlider(Qt.Orientation.Horizontal)
        s.setRange(lo, hi)
        s.setValue(val)
        s.setFixedWidth(180)
        s.valueChanged.connect(cb)
        return s

    # -- slots -------------------------------------------------------------

    def _on_effect(self, i: int) -> None:
        self.gl.effect = int(i)
        self.gl.update()

    def _on_light(self, i: int) -> None:
        self.gl.set_matcap(LIGHTINGS[int(i)])

    def _on_lo(self, v: int) -> None:
        if v > self._hi.value() - _THRESH_GAP:
            self._hi.blockSignals(True)
            self._hi.setValue(min(1000, v + _THRESH_GAP))
            self._hi.blockSignals(False)
            self.gl.thresh_hi = self._hi.value() / 1000.0
        self.gl.thresh_lo = v / 1000.0
        self.gl.update()

    def _on_hi(self, v: int) -> None:
        if v < self._lo.value() + _THRESH_GAP:
            self._lo.blockSignals(True)
            self._lo.setValue(max(0, v - _THRESH_GAP))
            self._lo.blockSignals(False)
            self.gl.thresh_lo = self._lo.value() / 1000.0
        self.gl.thresh_hi = v / 1000.0
        self.gl.update()

    def _on_den(self, v: int) -> None:
        self.gl.density = v / 100.0
        self.gl.update()

    def _on_bright(self, v: int) -> None:
        self.gl.brighten = v / 100.0
        self.gl.update()

    def _on_surf(self, v: int) -> None:
        self.gl.surface = v / 100.0
        self.gl.update()

    def _on_q(self, v: int) -> None:
        self.gl.steps = int(v)
        self.gl.update()

    def _on_clip_change(self, *_args) -> None:
        self._apply_clip()

    def _apply_clip(self) -> None:
        on = self._clip_enable.isChecked()
        for w in (self._az, self._el, self._depth, self._thick):
            w.setEnabled(on)
        if not on:
            self.gl.clip_active = 0
        else:
            self.gl.clip_active = 1
            self.gl.clip_normal = clip_normal_from(self._az.value(), self._el.value())
            self.gl.clip_depth = (self._depth.value() / 1000.0 - 0.5) * 1.8
            self.gl.clip_thick = 0.02 + (self._thick.value() / 1000.0) * 2.98
        self.gl.update()

    def reset_params(self) -> None:
        """Restore every control (and the GL state it drives) to defaults."""
        specs = (
            (self._effect.setCurrentIndex, DEFAULTS["effect"]),
            (self._light.setCurrentIndex, DEFAULTS["light"]),
            (self._lo.setValue, DEFAULTS["lo"]),
            (self._hi.setValue, DEFAULTS["hi"]),
            (self._den.setValue, DEFAULTS["den"]),
            (self._bright.setValue, DEFAULTS["bright"]),
            (self._surf.setValue, DEFAULTS["surf"]),
            (self._q.setValue, DEFAULTS["q"]),
            (self._az.setValue, DEFAULTS["az"]),
            (self._el.setValue, DEFAULTS["el"]),
            (self._depth.setValue, DEFAULTS["depth"]),
            (self._thick.setValue, DEFAULTS["thick"]),
            (self._clip_enable.setChecked, DEFAULTS["clip"]),
        )
        for setter, val in specs:
            setter(val)
        self._apply_clip()
        self.gl.reset_view()


# --------------------------------------------------------------------------
# Composite view: canvas + vertical controls (pure-3-D mode)
# --------------------------------------------------------------------------

class Nifti3DView(QWidget):
    """GL canvas + a vertical controls panel (scrollable), the pane's "3D" mode."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("pane-dark")
        self.gl = RaycastGLWidget()
        self.controls = Nifti3DControls(self.gl, vertical=True)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(self.controls)
        scroll.setFixedWidth(224)

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(2)
        root.addWidget(self.gl, 1)
        root.addWidget(scroll)

    def set_volume(self, vol: np.ndarray, spacing: tuple[float, float, float]) -> None:
        self.gl.set_volume_float(vol, spacing)

    def clear(self) -> None:
        self.gl.clear()

    def reset_view(self) -> None:
        self.gl.reset_view()


__all__ = [
    "Nifti3DControls",
    "Nifti3DView",
    "RaycastGLWidget",
    "request_gl_format",
    "normalize_to_u8",
    "clip_normal_from",
    "EFFECTS",
    "LIGHTINGS",
    "DEFAULTS",
]
