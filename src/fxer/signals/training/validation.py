"""Cross-validation splitters for time-series data."""

import logging
from itertools import combinations

import numpy as np

from fxer.core.exceptions import InsufficientDataError

logger = logging.getLogger(__name__)


class WalkForwardSplitter:
    """Walk-forward validation with expanding training windows.

    Creates chronologically ordered train/test splits with a 70/30 ratio
    and a purge gap between them to prevent look-ahead leakage.

    Each successive fold uses a larger training set while the test set
    slides forward in time.
    """

    def __init__(
        self,
        n_splits: int = 5,
        train_ratio: float = 0.70,
        purge_gap: int = 1,
    ) -> None:
        """
        Args:
            n_splits: Number of walk-forward folds.
            train_ratio: Fraction of available data for training in each fold.
            purge_gap: Number of samples to skip between train and test
                       to prevent information leakage.
        """
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        if not 0.5 <= train_ratio < 1.0:
            raise ValueError("train_ratio must be in [0.5, 1.0)")
        self.n_splits = n_splits
        self.train_ratio = train_ratio
        self.purge_gap = purge_gap

    def split(
        self, n_samples: int
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Generate walk-forward train/test index splits.

        Args:
            n_samples: Total number of samples.

        Returns:
            List of (train_indices, test_indices) tuples.
        """
        # Minimum test size per fold
        min_test = max(10, n_samples // (self.n_splits * 3))
        min_total = min_test * self.n_splits + self.purge_gap * self.n_splits + 50
        if n_samples < min_total:
            raise InsufficientDataError(
                f"Need at least {min_total} samples for {self.n_splits} folds, "
                f"got {n_samples}",
                required=min_total,
                available=n_samples,
            )

        indices = np.arange(n_samples)
        splits: list[tuple[np.ndarray, np.ndarray]] = []

        # Divide the data into n_splits + 1 segments for expanding window
        segment_size = n_samples // (self.n_splits + 1)

        for i in range(self.n_splits):
            # Training window expands: first 1..n_splits segments
            train_end = segment_size * (i + 1)
            test_start = train_end + self.purge_gap
            test_end = min(test_start + segment_size, n_samples)

            if test_start >= n_samples or test_end <= test_start:
                break

            train_idx = indices[:train_end]
            test_idx = indices[test_start:test_end]
            splits.append((train_idx, test_idx))

            logger.debug(
                "Fold %d: train[0:%d] test[%d:%d]",
                i + 1,
                train_end,
                test_start,
                test_end,
            )

        return splits


class CPCVSplitter:
    """Combinatorial Purged Cross-Validation (CPCV).

    Implements the CPCV method for measuring the Probability of
    Backtest Overfitting (PBO). Splits data into k groups, then
    takes all C(k, k-p) combinations for training/testing.

    A purge gap and embargo buffer prevent information leakage
    between adjacent groups.
    """

    def __init__(
        self,
        n_groups: int = 5,
        n_test_groups: int = 2,
        purge_gap: int = 1,
        embargo_pct: float = 0.01,
    ) -> None:
        """
        Args:
            n_groups: Number of time-based groups (k).
            n_test_groups: Number of groups reserved for testing (p).
            purge_gap: Samples to purge at train/test boundary.
            embargo_pct: Fraction of data to embargo after each test group.
        """
        if n_test_groups >= n_groups:
            raise ValueError("n_test_groups must be < n_groups")
        self.n_groups = n_groups
        self.n_test_groups = n_test_groups
        self.purge_gap = purge_gap
        self.embargo_pct = embargo_pct

    def split(
        self, n_samples: int
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Generate CPCV train/test index splits.

        Args:
            n_samples: Total number of samples.

        Returns:
            List of (train_indices, test_indices) tuples for each combination.
        """
        group_size = n_samples // self.n_groups
        embargo_size = max(1, int(n_samples * self.embargo_pct))

        if group_size < 10:
            raise InsufficientDataError(
                f"Groups too small ({group_size}). Need at least "
                f"{10 * self.n_groups} samples.",
                required=10 * self.n_groups,
                available=n_samples,
            )

        # Create group boundaries
        group_bounds: list[tuple[int, int]] = []
        for g in range(self.n_groups):
            start = g * group_size
            end = (g + 1) * group_size if g < self.n_groups - 1 else n_samples
            group_bounds.append((start, end))

        indices = np.arange(n_samples)
        splits: list[tuple[np.ndarray, np.ndarray]] = []

        # All combinations of n_test_groups out of n_groups for testing
        for test_group_indices in combinations(range(self.n_groups), self.n_test_groups):
            test_set: set[int] = set()
            purge_set: set[int] = set()

            for tg in test_group_indices:
                gs, ge = group_bounds[tg]
                test_set.update(range(gs, ge))

                # Purge: remove samples near boundaries
                purge_start = max(0, gs - self.purge_gap)
                purge_end = min(n_samples, ge + embargo_size)
                purge_set.update(range(purge_start, gs))
                purge_set.update(range(ge, purge_end))

            train_set = set(range(n_samples)) - test_set - purge_set
            train_idx = np.array(sorted(train_set), dtype=np.int64)
            test_idx = np.array(sorted(test_set), dtype=np.int64)

            if len(train_idx) > 0 and len(test_idx) > 0:
                splits.append((train_idx, test_idx))

        logger.info("CPCV generated %d splits from %d groups", len(splits), self.n_groups)
        return splits
