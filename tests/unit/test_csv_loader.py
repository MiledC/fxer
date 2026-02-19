"""Tests for the CSV data loader."""

from __future__ import annotations

import textwrap
from datetime import datetime
from pathlib import Path

import pytest

from fxer.core.types import RawBar
from fxer.data.loaders.base import DataLoader, LoadStats
from fxer.data.loaders.csv_loader import ColumnMapping, CSVLoader

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
SAMPLE_CSV = FIXTURES_DIR / "sample_bars.csv"


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _write_csv(tmp_path: Path, content: str, filename: str = "test.csv") -> Path:
    """Write *content* (dedented) to a temp CSV file and return its path."""
    p = tmp_path / filename
    p.write_text(textwrap.dedent(content).lstrip())
    return p


# ------------------------------------------------------------------ #
# Protocol conformance
# ------------------------------------------------------------------ #

class TestProtocolConformance:
    def test_csv_loader_satisfies_protocol(self):
        assert isinstance(CSVLoader(), DataLoader)


# ------------------------------------------------------------------ #
# Loading the fixture file
# ------------------------------------------------------------------ #

class TestLoadSampleCSV:
    @pytest.fixture()
    def loader(self) -> CSVLoader:
        loader = CSVLoader()
        loader.load(SAMPLE_CSV, symbol="XAUUSD")
        return loader

    def test_stats_row_count(self, loader: CSVLoader):
        stats = loader._stats
        assert stats is not None
        assert stats.rows_loaded == 60
        assert stats.rows_skipped == 0
        assert stats.errors == []
        assert stats.total_rows == 60

    def test_date_range(self, loader: CSVLoader):
        dr = loader.get_date_range()
        assert dr is not None
        start, end = dr
        assert start == datetime(2024, 1, 2, 0, 0, 0)
        assert end == datetime(2024, 1, 2, 22, 55, 0)

    def test_iter_bars_yields_raw_bars(self, loader: CSVLoader):
        bars = list(loader.iter_bars())
        assert len(bars) == 60
        assert all(isinstance(b, RawBar) for b in bars)

    def test_bars_chronological(self, loader: CSVLoader):
        bars = list(loader.iter_bars())
        timestamps = [b.timestamp for b in bars]
        assert timestamps == sorted(timestamps)

    def test_symbol_assigned(self, loader: CSVLoader):
        bars = list(loader.iter_bars())
        assert all(b.symbol == "XAUUSD" for b in bars)

    def test_first_bar_values(self, loader: CSVLoader):
        bar = next(loader.iter_bars())
        assert bar.timestamp == datetime(2024, 1, 2, 0, 0, 0)
        assert bar.open == "2062.50"
        assert bar.high == "2063.20"
        assert bar.low == "2061.80"
        assert bar.close == "2062.90"
        assert bar.volume == "1250"

    def test_load_returns_stats(self):
        loader = CSVLoader()
        stats = loader.load(SAMPLE_CSV, symbol="XAUUSD")
        assert isinstance(stats, LoadStats)
        assert stats.rows_loaded == 60


# ------------------------------------------------------------------ #
# Custom column mapping
# ------------------------------------------------------------------ #

class TestColumnMapping:
    def test_custom_mapping(self, tmp_path: Path):
        csv = _write_csv(tmp_path, """\
            Date,Open,High,Low,Close,Vol
            2024-01-01 00:00:00,100.0,101.0,99.0,100.5,500
            2024-01-01 00:05:00,100.5,102.0,100.0,101.5,600
        """)
        mapping = ColumnMapping(
            datetime="Date", open="Open", high="High",
            low="Low", close="Close", volume="Vol",
        )
        loader = CSVLoader(column_mapping=mapping)
        stats = loader.load(csv, symbol="TEST")
        assert stats.rows_loaded == 2
        bar = next(loader.iter_bars())
        assert bar.open == "100.0"

    def test_missing_volume_column_ok(self, tmp_path: Path):
        csv = _write_csv(tmp_path, """\
            datetime,open,high,low,close
            2024-01-01 00:00:00,100.0,101.0,99.0,100.5
        """)
        loader = CSVLoader()
        stats = loader.load(csv, symbol="TEST")
        assert stats.rows_loaded == 1
        bar = next(loader.iter_bars())
        assert bar.volume is None


# ------------------------------------------------------------------ #
# Datetime format handling
# ------------------------------------------------------------------ #

class TestDatetimeFormats:
    @pytest.mark.parametrize("dt_string,expected", [
        ("2024-01-15 10:30:00", datetime(2024, 1, 15, 10, 30, 0)),
        ("2024-01-15 10:30", datetime(2024, 1, 15, 10, 30)),
        ("2024-01-15T10:30:00", datetime(2024, 1, 15, 10, 30, 0)),
        ("2024-01-15T10:30:00Z", datetime(2024, 1, 15, 10, 30, 0)),
        ("2024/01/15 10:30:00", datetime(2024, 1, 15, 10, 30, 0)),
        ("01/15/2024 10:30:00", datetime(2024, 1, 15, 10, 30, 0)),
        ("15.01.2024 10:30:00", datetime(2024, 1, 15, 10, 30, 0)),
    ])
    def test_various_formats(self, tmp_path: Path, dt_string: str, expected: datetime):
        csv = _write_csv(tmp_path, f"""\
            datetime,open,high,low,close,volume
            {dt_string},100.0,101.0,99.0,100.5,500
        """)
        loader = CSVLoader()
        loader.load(csv, symbol="TEST")
        bar = next(loader.iter_bars())
        assert bar.timestamp == expected


# ------------------------------------------------------------------ #
# Error handling
# ------------------------------------------------------------------ #

class TestErrorHandling:
    def test_file_not_found(self):
        loader = CSVLoader()
        with pytest.raises(FileNotFoundError, match="CSV file not found"):
            loader.load("/nonexistent/path.csv", symbol="X")

    def test_missing_required_columns(self, tmp_path: Path):
        csv = _write_csv(tmp_path, """\
            date,price
            2024-01-01,100
        """)
        loader = CSVLoader()
        with pytest.raises(ValueError, match="missing required columns"):
            loader.load(csv, symbol="X")

    def test_empty_header(self, tmp_path: Path):
        p = tmp_path / "empty.csv"
        p.write_text("")
        loader = CSVLoader()
        with pytest.raises(ValueError, match="no header row"):
            loader.load(p, symbol="X")

    def test_malformed_rows_skipped(self, tmp_path: Path):
        csv = _write_csv(tmp_path, """\
            datetime,open,high,low,close,volume
            2024-01-01 00:00:00,100.0,101.0,99.0,100.5,500
            not-a-date,BAD,BAD,BAD,BAD,BAD
            2024-01-01 00:10:00,101.0,102.0,100.0,101.5,600
        """)
        loader = CSVLoader()
        stats = loader.load(csv, symbol="TEST")
        assert stats.rows_loaded == 2
        assert stats.rows_skipped == 1
        assert len(stats.errors) == 1
        assert "Line 3" in stats.errors[0]

    def test_malformed_datetime_skipped(self, tmp_path: Path):
        csv = _write_csv(tmp_path, """\
            datetime,open,high,low,close,volume
            GARBAGE,100.0,101.0,99.0,100.5,500
        """)
        loader = CSVLoader()
        stats = loader.load(csv, symbol="TEST")
        assert stats.rows_loaded == 0
        assert stats.rows_skipped == 1


# ------------------------------------------------------------------ #
# Edge cases
# ------------------------------------------------------------------ #

class TestEdgeCases:
    def test_empty_data_after_header(self, tmp_path: Path):
        csv = _write_csv(tmp_path, """\
            datetime,open,high,low,close,volume
        """)
        loader = CSVLoader()
        stats = loader.load(csv, symbol="X")
        assert stats.rows_loaded == 0
        assert loader.get_date_range() is None

    def test_single_row(self, tmp_path: Path):
        csv = _write_csv(tmp_path, """\
            datetime,open,high,low,close,volume
            2024-06-01 12:00:00,2000.0,2001.0,1999.0,2000.5,100
        """)
        loader = CSVLoader()
        stats = loader.load(csv, symbol="XAUUSD")
        assert stats.rows_loaded == 1
        dr = loader.get_date_range()
        assert dr is not None
        assert dr[0] == dr[1]

    def test_reload_replaces_previous_data(self, tmp_path: Path):
        csv1 = _write_csv(tmp_path, """\
            datetime,open,high,low,close,volume
            2024-01-01 00:00:00,100,101,99,100.5,10
        """, filename="a.csv")
        csv2 = _write_csv(tmp_path, """\
            datetime,open,high,low,close,volume
            2024-02-01 00:00:00,200,201,199,200.5,20
            2024-02-01 00:05:00,201,202,200,201.5,30
        """, filename="b.csv")
        loader = CSVLoader()
        loader.load(csv1, symbol="A")
        assert len(list(loader.iter_bars())) == 1

        loader.load(csv2, symbol="B")
        bars = list(loader.iter_bars())
        assert len(bars) == 2
        assert bars[0].symbol == "B"

    def test_iter_bars_before_load_yields_nothing(self):
        loader = CSVLoader()
        assert list(loader.iter_bars()) == []

    def test_get_date_range_before_load_returns_none(self):
        loader = CSVLoader()
        assert loader.get_date_range() is None
