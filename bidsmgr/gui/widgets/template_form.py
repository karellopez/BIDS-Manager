"""Turn a template field into a control, and back into a value.

The one place in the GUI that knows how a BIDS field becomes a widget. Both
surfaces that ask for metadata render through it: the dataset dialog, which
edits a whole dataset's template, and the inspection table's properties panel,
which edits one recording. Because the mapping lives here, neither of them
contains a BIDS field name, and a field added to a future schema version appears
in both without a line of code.

The control follows from what the schema declares about the field, never from a
list kept here:

===========================  ==========================================
the schema says              the user gets
===========================  ==========================================
a controlled vocabulary      an editable combo box, values preloaded
``boolean``                  unset / true / false, because a checkbox
                             cannot say "not stated"
``number`` / ``integer``     a numeric box with the unit beside it
``array`` of strings         a list editor, one row per entry
``object``                   key and value rows
anything else                a line edit with a shaped placeholder
===========================  ==========================================

One field is special by name rather than by type, and only one: ``Authors`` is a
list of people written "Family, Given". No separator survives that in free text,
so it gets a row per person with the two parts apart.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QDoubleValidator
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QFormLayout,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ...metadata.template_plan import REGION_AGNOSTIC
from ..theme_manager import CUR

# Fields that are lists of people. The only place a BIDS name appears in the
# GUI, and it earns it: "Lopez, Karel" is one author, so neither a comma nor a
# newline can be used to separate entries.
PEOPLE_FIELDS: frozenset[str] = frozenset({"Authors"})

_UNSET = "(not stated)"


class _RowList(QWidget):
    """A list the user grows a row at a time.

    ``make_row`` builds one row's widgets and returns ``(widget, read, write)``,
    so the same list serves plain strings and two-part names without knowing
    which it is holding.
    """

    def __init__(
        self,
        make_row: Callable[[], tuple[QWidget, Callable[[], Any], Callable[[Any], None]]],
        add_label: str = "Add",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._make_row = make_row
        self._rows: list[tuple[QWidget, Callable[[], Any], Callable[[Any], None]]] = []

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(2)
        self._rows_box = QVBoxLayout()
        self._rows_box.setSpacing(2)
        outer.addLayout(self._rows_box)

        add = QPushButton(add_label)
        add.setObjectName("template-add-row")
        add.clicked.connect(lambda: self.add_row())
        outer.addWidget(add, 0, Qt.AlignmentFlag.AlignLeft)
        self.add_row()

    def add_row(self, value: Any = None) -> None:
        widget, read, write = self._make_row()
        if value is not None:
            write(value)

        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(widget, 1)
        remove = QPushButton("−")
        remove.setFixedWidth(26)
        remove.setToolTip("Remove this entry")
        layout.addWidget(remove, 0)
        self._rows_box.addWidget(row)

        entry = (row, read, write)
        self._rows.append(entry)

        def drop() -> None:
            if entry in self._rows:
                self._rows.remove(entry)
            row.setParent(None)
            if not self._rows:
                self.add_row()

        remove.clicked.connect(drop)

    def value(self) -> list:
        out = []
        for _row, read, _write in self._rows:
            item = read()
            if item not in (None, "", [], {}):
                out.append(item)
        return out

    def set_value(self, values) -> None:
        for row, _read, _write in list(self._rows):
            row.setParent(None)
        self._rows.clear()
        for item in values or []:
            self.add_row(item)
        if not self._rows:
            self.add_row()


def _string_row() -> tuple[QWidget, Callable[[], Any], Callable[[Any], None]]:
    edit = QLineEdit()
    return edit, lambda: edit.text().strip(), lambda v: edit.setText(str(v))


def _person_row() -> tuple[QWidget, Callable[[], Any], Callable[[Any], None]]:
    """Family and given name apart, composed the way BIDS writes them."""
    row = QWidget()
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(4)
    family = QLineEdit()
    family.setPlaceholderText("Family name")
    given = QLineEdit()
    given.setPlaceholderText("Given name")
    layout.addWidget(family, 3)
    layout.addWidget(given, 2)

    def read() -> str:
        f, g = family.text().strip(), given.text().strip()
        if f and g:
            return f"{f}, {g}"
        return f or g

    def write(value: Any) -> None:
        f, _, g = str(value).partition(",")
        family.setText(f.strip())
        given.setText(g.strip())

    return row, read, write


def build_field_widget(field) -> QWidget:
    """The control for one :class:`TemplateField`."""
    if field.name in PEOPLE_FIELDS:
        widget = _RowList(_person_row, add_label="Add author")
        widget.setProperty("template_kind", "people")
        return widget

    if field.enum:
        combo = QComboBox()
        combo.setEditable(True)
        combo.addItem("")
        combo.addItems([str(v) for v in field.enum])
        combo.setProperty("template_kind", "enum")
        return combo

    if field.type == "boolean":
        combo = QComboBox()
        combo.addItems([_UNSET, "true", "false"])
        combo.setProperty("template_kind", "boolean")
        return combo

    if field.type == "array" and field.item_type in ("", "string"):
        widget = _RowList(_string_row, add_label="Add entry")
        widget.setProperty("template_kind", "list")
        return widget

    edit = QLineEdit()
    edit.setPlaceholderText(field.placeholder)
    if field.type in ("number", "integer"):
        edit.setValidator(QDoubleValidator())
        edit.setProperty("template_kind", "number")
    else:
        edit.setProperty("template_kind", "text")
    return edit


def read_field_widget(widget: QWidget, field) -> Any:
    """The widget's value, in the shape the schema declares. ``None`` if unset."""
    kind = widget.property("template_kind")

    if kind in ("people", "list"):
        values = widget.value()
        return values or None

    if kind == "boolean":
        text = widget.currentText()
        return None if text == _UNSET else text == "true"

    text = (
        widget.currentText() if isinstance(widget, QComboBox) else widget.text()
    ).strip()
    if not text:
        return None
    if kind == "number":
        try:
            return int(text) if field.type == "integer" else float(text)
        except ValueError:
            return text
    if field.type == "array":
        # An array whose items are not strings: keep what was typed rather than
        # guessing a split that may be wrong.
        return [text]
    return text


def write_field_widget(widget: QWidget, value: Any) -> None:
    """Show ``value`` in the widget, whatever kind it is."""
    kind = widget.property("template_kind")
    if kind in ("people", "list"):
        widget.set_value(value if isinstance(value, list) else [value])
        return
    if kind == "boolean":
        widget.setCurrentText(
            _UNSET if value is None else ("true" if value else "false")
        )
        return
    text = (
        ", ".join(str(v) for v in value) if isinstance(value, list) else str(value)
    )
    if isinstance(widget, QComboBox):
        widget.setCurrentText(text)
    else:
        widget.setText(text)


def field_label(field, *, colour: bool = True) -> str:
    """The label for a field: its name, its level, and its unit.

    The level is shown as a mark AND, optionally, a colour. The mark stays when
    colour is off, so the information does not depend on being able to see it.
    """
    mark = {"required": " *", "recommended": " ·"}.get(field.level, "")
    unit = f" ({field.unit})" if field.unit else ""
    if not colour:
        return f"{field.name}{mark}{unit}:"
    pal = CUR()
    tone = {
        "required": pal["error"],
        "recommended": pal["warning"],
        "deprecated": pal["muted"],
    }.get(field.level, pal["text"])
    return (
        f'<span style="color:{tone};">{field.name}{mark}</span>'
        f'<span style="color:{pal["muted"]};">{unit}</span>:'
    )


def level_legend(*, colour: bool = True) -> QLabel:
    """A one-line key to the marks, so nobody has to infer them."""
    pal = CUR()
    if colour:
        text = (
            f'<span style="color:{pal["error"]};">* required</span> &nbsp; '
            f'<span style="color:{pal["warning"]};">· recommended</span> &nbsp; '
            f'<span style="color:{pal["muted"]};">optional</span>'
        )
    else:
        text = "* required &nbsp; · recommended &nbsp; (unmarked) optional"
    label = QLabel(text)
    label.setTextFormat(Qt.TextFormat.RichText)
    label.setStyleSheet("background: transparent;")
    return label


# ---------------------------------------------------------------------------
# Collapsible sections
# ---------------------------------------------------------------------------
#
# NOT PanelFrame, though the plan first said so. That widget collapses a docked
# pane to a 24px vertical strip and can detach it into its own window, which is
# right for the converter's panes and wrong inside a scrolling form: a folded
# section here should simply stop taking room, keeping its heading readable.


class CollapsibleSection(QWidget):
    """A titled section that folds away, with a count of what is inside.

    The heading stays legible when folded, and carries how many fields the
    section holds and how many of them BIDS requires, so a user can decide
    whether to open it without opening it.
    """

    def __init__(
        self,
        title: str,
        *,
        subtitle: str = "",
        badge: str = "",
        level: int = 0,
        expanded: bool = True,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._expanded = expanded

        outer = QVBoxLayout(self)
        outer.setContentsMargins(level * 10, 0, 0, 0)
        outer.setSpacing(2)

        pal = CUR()
        self._header = QPushButton()
        self._header.setObjectName("template-section-header")
        self._header.setCheckable(True)
        self._header.setChecked(expanded)
        self._header.setCursor(Qt.CursorShape.PointingHandCursor)
        self._header.setStyleSheet(
            "QPushButton{text-align:left; padding:4px 6px; border:none; "
            f"background: transparent; color:{pal['text']};}}"
            f"QPushButton:hover{{background:{pal['surface2']};}}"
        )
        self._title_text = title
        self._subtitle = subtitle
        self._badge = badge
        outer.addWidget(self._header)

        self._body = QWidget()
        self._body_layout = QVBoxLayout(self._body)
        self._body_layout.setContentsMargins(12, 0, 0, 6)
        self._body_layout.setSpacing(6)
        outer.addWidget(self._body)

        self._header.toggled.connect(self._on_toggled)
        self._refresh_header()
        self._body.setVisible(expanded)

    def _refresh_header(self) -> None:
        pal = CUR()
        caret = "▾" if self._expanded else "▸"
        parts = [f'<b>{self._title_text}</b>']
        if self._subtitle:
            parts.append(f'<span style="color:{pal["muted"]};">{self._subtitle}</span>')
        if self._badge:
            parts.append(f'<span style="color:{pal["muted"]};">{self._badge}</span>')
        self._header.setText(f"{caret}  " + "   ".join(
            _strip_markup(p) for p in parts
        ))
        self._header.setToolTip(self._subtitle or "")

    def _on_toggled(self, checked: bool) -> None:
        self._expanded = checked
        self._body.setVisible(checked)
        self._refresh_header()
        self.toggled_by_user.emit(checked)

    toggled_by_user = pyqtSignal(bool)

    def body(self) -> QVBoxLayout:
        return self._body_layout

    def add(self, widget: QWidget) -> None:
        self._body_layout.addWidget(widget)

    def set_expanded(self, expanded: bool) -> None:
        self._header.setChecked(expanded)

    def is_expanded(self) -> bool:
        return self._expanded


def _strip_markup(text: str) -> str:
    """A QPushButton shows plain text, so drop the tags but keep the words."""
    import re

    return re.sub(r"<[^>]+>", "", text)


__all__ = [
    "CollapsibleSection",
    "TemplateTree",
    "PEOPLE_FIELDS",
    "build_field_widget",
    "field_label",
    "level_legend",
    "read_field_widget",
    "write_field_widget",
]


# ---------------------------------------------------------------------------
# The tree, rendered
# ---------------------------------------------------------------------------


class TemplateTree(QWidget):
    """Renders a template tree: regions, modalities, then one node per file.

    The single widget both metadata surfaces use. It holds no BIDS knowledge of
    its own: what to ask, at what level, in what shape, all arrives in the nodes.

    ``values`` maps a node's storage key to the answers already stored, and
    :meth:`values_by_key` gives them back in the same shape, so a caller reads
    and writes without knowing a field name either.
    """

    def __init__(
        self,
        nodes,
        *,
        values: Optional[dict] = None,
        colour_levels: bool = True,
        collapsed_keys: Optional[set] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._colour = colour_levels
        self._widgets: dict[str, dict[str, QWidget]] = {}
        self._sections: dict[str, "CollapsibleSection"] = {}
        self._fields: dict[str, dict] = {}
        collapsed_keys = collapsed_keys or set()

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)
        outer.addWidget(level_legend(colour=colour_levels))
        for node in nodes:
            outer.addWidget(
                self._render(
                    node, values or {}, collapsed_keys, level=0,
                    # The agnostic region is what a user fills first, so it
                    # opens; the per-file boxes under a modality do not, or a
                    # four-modality dataset greets them with a wall.
                    open_files=(node.key == REGION_AGNOSTIC),
                )
            )
        outer.addStretch(1)

    # -- building ------------------------------------------------------

    def _render(
        self, node, values: dict, collapsed: set, level: int,
        open_files: bool = False,
    ) -> QWidget:
        badge = ""
        if node.field_count:
            badge = f"{node.field_count} fields"
            if node.required_count:
                badge += f", {node.required_count} required"

        section = CollapsibleSection(
            node.label,
            subtitle=node.subtitle,
            badge=badge,
            level=level,
            # A file's own fields start folded: a dataset with four modalities
            # would otherwise open as a wall of boxes. The groupings above stay
            # open, so what exists is visible at a glance.
            expanded=(node.key not in collapsed)
            and (node.kind != "file" or open_files),
        )
        self._sections[node.key] = section

        if node.is_leaf:
            section.add(self._render_fields(node, values.get(node.key, {})))
        for child in node.children:
            section.add(
                self._render(child, values, collapsed, level + 1, open_files)
            )
        return section

    def _render_fields(self, node, stored: dict) -> QWidget:
        holder = QWidget()
        form = QFormLayout(holder)
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(4)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        widgets: dict[str, QWidget] = {}
        fields: dict[str, object] = {}
        for field in node.section.fields:
            widget = build_field_widget(field)
            if field.description:
                widget.setToolTip(field.description)
            if field.name in stored:
                write_field_widget(widget, stored[field.name])

            label = QLabel(field_label(field, colour=self._colour))
            label.setTextFormat(Qt.TextFormat.RichText)
            if field.description:
                label.setToolTip(field.description)
            form.addRow(label, widget)
            widgets[field.name] = widget
            fields[field.name] = field

        self._widgets[node.key] = widgets
        self._fields[node.key] = fields
        return holder

    # -- reading -------------------------------------------------------

    def values_by_key(self) -> dict[str, dict]:
        """What the user has typed, per node, as the schema shapes it."""
        out: dict[str, dict] = {}
        for key, widgets in self._widgets.items():
            answers = {}
            for name, widget in widgets.items():
                value = read_field_widget(widget, self._fields[key][name])
                if value not in (None, "", [], {}):
                    answers[name] = value
            if answers:
                out[key] = answers
        return out

    def collapsed_keys(self) -> set:
        """Which sections the user folded, to restore next time."""
        return {
            key for key, section in self._sections.items()
            if not section.is_expanded()
        }
