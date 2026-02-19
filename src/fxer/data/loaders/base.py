"""Base protocol and types for data loaders."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Generator, Protocol, runtime_checkable

from fxer.core.types import RawBar


@dataclass(frozen=True, slots=True)
class LoadStats:
    """Statistics from a data load operation."""

    rows_loaded: int
    rows_skipped: int
    start_date: datetime | None
    end_date: datetime | None
    errors: list[str] = field(default_factory=list)

    @property
    def total_rows(self) -> int:
        return self.rows_loaded + self.rows_skipped


@runtime_checkable
class DataLoader(Protocol):
    """Protocol for all data loaders (CSV, broker feeds, etc.)."""

    def load(self, path: Path | str, symbol: str) -> LoadStats:
        """Load data from the given path for the specified symbol.

        Args:
            path: File path or URI to the data source.
            symbol: Trading symbol (e.g. "XAUUSD").

        Returns:
            LoadStats with information about the load operation.
        """
        ...

    def iter_bars(self) -> Generator[RawBar, None, None]:
        """Iterate over loaded bars in chronological order.

        Yields:
            RawBar instances sorted by timestamp ascending.
        """
        ...

    def get_date_range(self) -> tuple[datetime, datetime] | None:
        """Return the date range of loaded data.

        Returns:
            Tuple of (start_date, end_date) or None if no data is loaded.
        """
        ...
