from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QWidget

from xiangqi_agent.domain.board import BoardState, Move, Orientation

_PIECE_NAMES = {
    "K": "帅",
    "A": "仕",
    "B": "相",
    "N": "马",
    "R": "车",
    "C": "炮",
    "P": "兵",
    "k": "将",
    "a": "士",
    "b": "象",
    "n": "马",
    "r": "车",
    "c": "炮",
    "p": "卒",
}


class BoardWidget(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._board: BoardState | None = None
        self._last_move: Move | None = None
        self.setMinimumSize(430, 520)

    @property
    def board(self) -> BoardState | None:
        return self._board

    @property
    def last_move(self) -> Move | None:
        return self._last_move

    def set_board(self, board: BoardState, *, last_move: Move | None = None) -> None:
        self._board = board
        self._last_move = last_move
        self.update()

    def sizeHint(self) -> QSize:
        return QSize(520, 650)

    def paintEvent(self, _event: object) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#f4dfb3"))
        margin = min(self.width() * 0.085, self.height() * 0.065)
        board_width = self.width() - 2 * margin
        board_height = self.height() - 2 * margin
        dx = board_width / 8
        dy = board_height / 9
        grid_pen = QPen(QColor("#7b542d"), 1.5)
        painter.setPen(grid_pen)

        for row in range(10):
            y = margin + row * dy
            painter.drawLine(QPointF(margin, y), QPointF(margin + board_width, y))
        for column in range(9):
            x = margin + column * dx
            if column in (0, 8):
                painter.drawLine(QPointF(x, margin), QPointF(x, margin + board_height))
            else:
                painter.drawLine(QPointF(x, margin), QPointF(x, margin + 4 * dy))
                painter.drawLine(
                    QPointF(x, margin + 5 * dy), QPointF(x, margin + board_height)
                )

        for top in (0, 7):
            painter.drawLine(
                QPointF(margin + 3 * dx, margin + top * dy),
                QPointF(margin + 5 * dx, margin + (top + 2) * dy),
            )
            painter.drawLine(
                QPointF(margin + 5 * dx, margin + top * dy),
                QPointF(margin + 3 * dx, margin + (top + 2) * dy),
            )

        painter.setPen(QColor("#7b542d"))
        river_font = QFont("Microsoft YaHei", max(12, round(dy * 0.28)))
        river_font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 8)
        painter.setFont(river_font)
        river = QRectF(margin, margin + 4 * dy, board_width, dy)
        painter.drawText(river, Qt.AlignmentFlag.AlignCenter, "楚 河        汉 界")

        if self._board is None:
            return
        if self._last_move is not None:
            highlight_radius = min(dx, dy) * 0.48
            painter.setPen(QPen(Qt.PenStyle.NoPen))
            painter.setBrush(QColor(244, 197, 66, 150))
            for index in (self._last_move.from_index, self._last_move.to_index):
                row, column = divmod(index, 9)
                if self._board.orientation == Orientation.BLACK_BOTTOM:
                    row, column = 9 - row, 8 - column
                painter.drawEllipse(
                    QPointF(margin + column * dx, margin + row * dy),
                    highlight_radius,
                    highlight_radius,
                )
        radius = min(dx, dy) * 0.39
        piece_font = QFont("Microsoft YaHei", max(13, round(radius * 0.88)), QFont.Weight.Bold)
        painter.setFont(piece_font)
        for index, piece in enumerate(self._board.pieces):
            if piece == ".":
                continue
            row, column = divmod(index, 9)
            if self._board.orientation == Orientation.BLACK_BOTTOM:
                row, column = 9 - row, 8 - column
            center = QPointF(margin + column * dx, margin + row * dy)
            painter.setBrush(QColor("#f8e2aa"))
            painter.setPen(QPen(QColor("#9b672d"), 2))
            painter.drawEllipse(center, radius, radius)
            painter.setPen(QColor("#a52720") if piece.isupper() else QColor("#263238"))
            painter.drawText(
                QRectF(center.x() - radius, center.y() - radius, radius * 2, radius * 2),
                Qt.AlignmentFlag.AlignCenter,
                _PIECE_NAMES[piece],
            )
