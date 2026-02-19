"""CSV data loader implementation."""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Generator

from fxer.core.types import RawBar
from fxer.data.loaders.base import DataLoader, LoadStats

logger = logging.getLogger(__name__)

# Common datetime formats for forex/commodity data
DEFAULT_DATETIME_FORMATS = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y/%m/%d %H:%M:%S",
    "%Y/%m/%d %H:%M",
    "%m/%d/%Y %H:%M:%S",
    "%m/%d/%Y %H:%M",
    "%d.%m.%Y %H:%M:%S",
    "%d.%m.%Y %H:%M",
)


@dataclass
class ColumnMapping:
    """Maps CSV column names to expected fields.

    Allows loading CSVs that use different header names, e.g.
    ``ColumnMapping(datetime="Date", open="Open", ...)`` for a file
    whose columns are Date,Open,High,Low,Close,Vol.
    """

    datetime: str = "datetime"
    open: str = "open"
    high: str = "high"
    low: str = "low"
    close: str = "close"
    volume: str = "volume"


@dataclass
class CSVLoader:
    """Load OHLCV bar data from CSV files.

    Supports configurable column mapping and multiple datetime formats.
    After calling :meth:`load`, iterate bars with :meth:`iter_bars`
    or inspect the date range with :meth:`get_date_range`.
    """

    column_mapping: ColumnMapping = field(default_factory=ColumnMapping)
    datetime_formats: tuple[str, ...] = DEFAULT_DATETIME_FORMATS

    _bars: list[RawBar] = field(default_factory=list, init=False, repr=False)
    _stats: LoadStats | None = field(default=None, init=False, repr=False)

    # ------------------------------------------------------------------ #
    # DataLoader protocol
    # ------------------------------------------------------------------ #

    def load(self, path: Path | str, symbol: str) -> LoadStats:
        """Load CSV file into memory, returning load statistics."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"CSV file not found: {path}")

        bars: list[RawBar] = []
        errors: list[str] = []
        rows_skipped = 0

        with path.open(newline="") as fh:
            reader = csv.DictReader(fh)

            if reader.fieldnames is None:
                raise ValueError(f"CSV file has no header row: {path}")

            self._validate_columns(reader.fieldnames)

            for line_num, row in enumerate(reader, start=2):
                try:
                    bar = self._parse_row(row, symbol)
                    bars.append(bar)
                except Exception as exc:
                    rows_skipped += 1
                    msg = f"Line {line_num}: {exc}"
                    errors.append(msg)
                    logger.debug("Skipping malformed row – %s", msg)

        bars.sort(key=lambda b: b.timestamp)
        self._bars = bars

        start = bars[0].timestamp if bars else None
        end = bars[-1].timestamp if bars else None

        self._stats = LoadStats(
            rows_loaded=len(bars),
            rows_skipped=rows_skipped,
            start_date=start,
            end_date=end,
            errors=errors,
        )
        return self._stats

    def iter_bars(self) -> Generator[RawBar, None, None]:
        """Yield bars in chronological order."""
        yield from self._bars

    def get_date_range(self) -> tuple[datetime, datetime] | None:
        """Return ``(start, end)`` timestamps, or ``None`` if empty."""
        if not self._bars:
            return None
        return (self._bars[0].timestamp, self._bars[-1].timestamp)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _validate_columns(self, fieldnames: list[str] | tuple[str, ...]) -> None:
        """Raise ``ValueError`` if required columns are missing."""
        cm = self.column_mapping
        required = {cm.datetime, cm.open, cm.high, cm.low, cm.close}
        present = {name.strip() for name in fieldnames}
        missing = required - present
        if missing:
            raise ValueError(
                f"CSV is missing required columns: {sorted(missing)}. "
                f"Found columns: {sorted(present)}"
            )

    def _parse_datetime(self, raw: str) -> datetime:
        """Try each configured format until one succeeds."""
        value = raw.strip()
        for fmt in self.datetime_formats:
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue
        raise ValueError(f"Cannot parse datetime '{value}' with any known format")

    def _parse_row(self, row: dict[str, str], symbol: str) -> RawBar:
        """Convert a single CSV dict-row into a RawBar."""
        cm = self.column_mapping
        timestamp = self._parse_datetime(row[cm.datetime])

        volume_raw = row.get(cm.volume, "").strip()
        volume = volume_raw if volume_raw else None

        return RawBar(
            timestamp=timestamp,
            open=row[cm.open].strip(),
            high=row[cm.high].strip(),
            low=row[cm.low].strip(),
            close=row[cm.close].strip(),
            volume=volume,
            symbol=symbol,
        )
