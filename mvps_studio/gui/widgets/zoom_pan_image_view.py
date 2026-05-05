"""
mvps_studio.gui.widgets.zoom_pan_image_view
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A QGraphicsView-based image viewer with:
  - Scroll wheel        → zoom in / out (anchored under cursor)
  - Left-button drag    → pan  (unrestricted — goes beyond scene rect)
  - Double-click        → fit to view
  - Ctrl+0              → fit to view
  - setPixmap(pm)       → display a QPixmap
  - setText(t)          → display a plain text placeholder

Panning is fully unrestricted: the scene rect is padded so the scrollbars
always have range, and dragging translates the viewport directly via
scrollbar offsets — never clamped by image dimensions.
"""

from PySide6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QGraphicsTextItem, QSizePolicy,
)
from PySide6.QtCore import Qt, QRectF, QPoint
from PySide6.QtGui import QPainter, QPixmap, QWheelEvent, QMouseEvent


# Extra padding (px) added around the image so the user can pan freely
_PAN_PAD = 4000


class ZoomPanImageView(QGraphicsView):
    """Drop-in replacement for ScaledImageLabel in carousel / detail views."""

    _ZOOM_FACTOR = 1.15

    def __init__(self, parent=None):
        self._scene = QGraphicsScene()           # keep Python ref — prevents GC
        super().__init__(self._scene, parent)

        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        # No built-in drag mode — we implement unrestricted panning ourselves
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setBackgroundBrush(Qt.GlobalColor.darkGray)

        self._pix_item = QGraphicsPixmapItem()
        self._pix_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self._scene.addItem(self._pix_item)

        self._text_item = QGraphicsTextItem()
        self._scene.addItem(self._text_item)

        self._pan_active = False
        self._last_pan: QPoint | None = None
        self._pixmap = QPixmap()

    # ------------------------------------------------------------------
    def setPixmap(self, pixmap: QPixmap):
        """Display a QPixmap. Fits it to the view on first load."""
        self._pixmap = pixmap
        self._text_item.setPlainText("")
        if pixmap and not pixmap.isNull():
            self._pix_item.setPixmap(pixmap)
            # Padded scene rect so panning is always unrestricted
            w, h = pixmap.width(), pixmap.height()
            self._scene.setSceneRect(
                QRectF(-_PAN_PAD, -_PAN_PAD, w + 2 * _PAN_PAD, h + 2 * _PAN_PAD))
            self._fit()
        else:
            self._pix_item.setPixmap(QPixmap())

    def setText(self, text: str):
        """Display a placeholder string when no image is available."""
        self._pix_item.setPixmap(QPixmap())
        self._text_item.setPlainText(text)
        br = self._text_item.boundingRect()
        self._scene.setSceneRect(
            QRectF(-_PAN_PAD, -_PAN_PAD,
                   br.width() + 2 * _PAN_PAD, br.height() + 2 * _PAN_PAD))
        self._fit()

    def _fit(self):
        self.resetTransform()
        # Fit just the image (not the whole padded scene rect)
        if not self._pixmap.isNull():
            img_rect = QRectF(self._pixmap.rect())
        else:
            img_rect = self._text_item.boundingRect()
        self.fitInView(img_rect, Qt.AspectRatioMode.KeepAspectRatio)

    # ------------------------------------------------------------------
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not self._pixmap.isNull():
            vr = self.viewport().rect()
            img_rect = QRectF(self._pixmap.rect())
            mapped = self.mapFromScene(img_rect).boundingRect()
            scale_x = vr.width()  / max(mapped.width(),  1)
            scale_y = vr.height() / max(mapped.height(), 1)
            # Auto-fit only if user hasn't zoomed in (scale ≈ fit scale)
            if min(scale_x, scale_y) > 0.95:
                self._fit()

    # ------------------------------------------------------------------
    # Panning — left-button drag, fully unrestricted
    # ------------------------------------------------------------------
    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self._pan_active = True
            self._last_pan = event.position().toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._pan_active and self._last_pan is not None:
            pos = event.position().toPoint()
            delta = pos - self._last_pan
            self._last_pan = pos
            # Move scrollbars directly — unrestricted by scene rect
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y())
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self._pan_active = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    # ------------------------------------------------------------------
    def wheelEvent(self, event: QWheelEvent):
        """Scroll = zoom (no modifier required).  Ctrl+Scroll → parent."""
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            event.ignore()
            return
        factor = self._ZOOM_FACTOR if event.angleDelta().y() > 0 else 1 / self._ZOOM_FACTOR
        self.scale(factor, factor)
        event.accept()

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        """Double-click resets to fit-to-view."""
        self._fit()
        event.accept()

    def keyPressEvent(self, event):
        """Ctrl+0 = fit to view; all other keys propagate to parent."""
        if event.key() == Qt.Key.Key_0 and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self._fit()
            event.accept()
        else:
            event.ignore()
