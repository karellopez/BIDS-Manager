"""A layout whose items wrap onto another row instead of forcing a width.

A ``QHBoxLayout`` reports a minimum width equal to the SUM of what its
children need, so a bar of eight controls cannot be made narrower than all
eight of them laid end to end. In a pane the user is meant to be able to
drag narrow, that sum is the floor: the Editor's sidecar could not go below
617 pixels, and the content it was there to show could have gone to 62.

This wraps instead. The minimum width becomes the WIDEST SINGLE item rather
than the sum of all of them, and the rows reflow as the pane is dragged, so
nothing is clipped and nothing is cut off the right-hand edge.

Qt ships no flow layout; this is the documented pattern, with three
departures worth naming:

* **Spacing is explicit, per axis.** Qt's example asks the style for it,
  which on this app returns a value tuned for dialogs and reads as a gap in
  a toolbar.
* **``heightForWidth`` is honoured**, which is what makes the wrapping
  real: a ``QBoxLayout`` parent asks for it and resizes the row. (A
  ``QFormLayout`` does NOT ask through a nested widget, which is a separate
  trap and the reason the scope bar's summary is elided rather than
  wrapped.)
* **Hidden items take no room.** The sidecar hides its field verbs in BIDS
  view, and a hidden button that still reserved its width would put the
  floor back.
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import QPoint, QRect, QSize, Qt
from PyQt6.QtWidgets import QLayout, QLayoutItem, QSizePolicy, QWidget


class FlowLayout(QLayout):
    """Lay items left to right, wrapping onto a new row when they run out."""

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        *,
        margin: int = 0,
        h_spacing: int = 8,
        v_spacing: int = 6,
    ) -> None:
        super().__init__(parent)
        self._items: list[QLayoutItem] = []
        self._h = h_spacing
        self._v = v_spacing
        self.setContentsMargins(margin, margin, margin, margin)

    # -- QLayout plumbing --------------------------------------------------

    def addItem(self, item: QLayoutItem) -> None:  # noqa: N802 - Qt naming
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int) -> Optional[QLayoutItem]:  # noqa: N802
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    def takeAt(self, index: int) -> Optional[QLayoutItem]:  # noqa: N802
        if 0 <= index < len(self._items):
            return self._items.pop(index)
        return None

    def expandingDirections(self) -> Qt.Orientation:  # noqa: N802
        return Qt.Orientation(0)

    # -- sizing ------------------------------------------------------------

    def hasHeightForWidth(self) -> bool:  # noqa: N802
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802
        return self._arrange(QRect(0, 0, width, 0), apply=False)

    def setGeometry(self, rect: QRect) -> None:  # noqa: N802
        super().setGeometry(rect)
        self._arrange(rect, apply=True)

    def sizeHint(self) -> QSize:  # noqa: N802
        """One row, everything on it. What the bar looks like with room."""
        left, top, right, bottom = self.getContentsMargins()
        width = height = 0
        visible = [i for i in self._items if not self._hidden(i)]
        for index, item in enumerate(visible):
            hint = item.sizeHint()
            width += hint.width() + (self._h if index else 0)
            height = max(height, hint.height())
        return QSize(width + left + right, height + top + bottom)

    def minimumSize(self) -> QSize:  # noqa: N802
        """The WIDEST ITEM, not the sum. This is the point of the class."""
        left, top, right, bottom = self.getContentsMargins()
        size = QSize(0, 0)
        for item in self._items:
            if self._hidden(item):
                continue
            size = size.expandedTo(item.minimumSize())
        return QSize(size.width() + left + right, size.height() + top + bottom)

    # -- the wrap ----------------------------------------------------------

    @staticmethod
    def _hidden(item: QLayoutItem) -> bool:
        widget = item.widget()
        return widget is not None and widget.isHidden()

    def _arrange(self, rect: QRect, *, apply: bool) -> int:
        """Place the items inside ``rect``; return the height they need."""
        left, top, right, bottom = self.getContentsMargins()
        area = rect.adjusted(left, top, -right, -bottom)
        x = area.x()
        y = area.y()
        row_height = 0

        for item in self._items:
            if self._hidden(item):
                continue
            hint = item.sizeHint()
            # An item wider than the row gets the row: clipping it would be
            # worse than letting it be the width the bar cannot go below,
            # which ``minimumSize`` already reports.
            width = min(hint.width(), max(area.width(), hint.width()))
            if row_height and x + width > area.right() + 1:
                x = area.x()
                y += row_height + self._v
                row_height = 0
            if apply:
                item.setGeometry(QRect(QPoint(x, y), QSize(width, hint.height())))
            x += width + self._h
            row_height = max(row_height, hint.height())

        return y + row_height - rect.y() + bottom


def flow(widget: QWidget, **kwargs) -> FlowLayout:
    """Give ``widget`` a :class:`FlowLayout` and let it shrink to one item.

    The size policy matters as much as the layout, and its names are the
    wrong way round from how they read:

    * **Horizontally ``Preferred``**, which means "the hint is my best
      width and I can do with less, down to my minimum". ``Minimum`` means
      the opposite of what it sounds like, "the hint is already the least I
      will take", and set there it pins the bar at its one-row width and
      the wrapping never happens.
    * **Vertically ``Minimum``**, so the bar may grow as rows are added and
      never shrinks into them.
    * **``setHeightForWidth(True)``**, or the parent never asks how tall
      the wrapped rows came out and the extra ones are clipped.
    """
    layout = FlowLayout(widget, **kwargs)
    policy = QSizePolicy(
        QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum,
    )
    policy.setHeightForWidth(True)
    widget.setSizePolicy(policy)
    return layout


__all__ = ["FlowLayout", "flow"]
