"""Unit tests for diffracc/utils/logger.py."""
import logging

import pytest

from diffracc.utils import logger as logger_module
from diffracc.utils.logger import LoggingLevels, get_logger, show_dl_progress


class TestLoggingLevels:
    """Tests for the LoggingLevels enum, which maps to standard logging levels."""

    def test_values_match_stdlib_logging_levels(self):
        """
        Test that the values of LoggingLevels enum match the corresponding standard logging levels in the logging
        module.
        """
        assert LoggingLevels.DEBUG.value == logging.DEBUG
        assert LoggingLevels.INFO.value == logging.INFO
        assert LoggingLevels.WARNING.value == logging.WARNING
        assert LoggingLevels.ERROR.value == logging.ERROR
        assert LoggingLevels.CRITICAL.value == logging.CRITICAL


class TestGetLogger:
    """Tests for the get_logger function, which returns a configured logger instance."""

    def test_returns_logger_with_given_name(self):
        """Test that get_logger returns a logger instance with the specified name."""
        log = get_logger("my.test.logger")
        assert log.name == "my.test.logger"

    def test_defaults_to_info_level(self):
        """Test that the default logging level is INFO if not explicitly specified."""
        log = get_logger("my.test.logger.default")
        assert log.level == logging.INFO

    def test_respects_explicit_level(self):
        """Test that get_logger respects an explicitly specified logging level."""
        log = get_logger("my.test.logger.debug", level=LoggingLevels.DEBUG.value)
        assert log.level == logging.DEBUG

    def test_adds_exactly_one_handler(self):
        """Test that get_logger adds exactly one StreamHandler to the logger, even if called multiple times."""
        log = get_logger("my.test.logger.single")
        assert len(log.handlers) == 1
        assert isinstance(log.handlers[0], logging.StreamHandler)

    def test_repeated_calls_with_same_name_do_not_accumulate_handlers(self):
        """Test that repeated calls to get_logger with the same name do not accumulate multiple handlers."""
        get_logger("my.test.logger.repeat")
        get_logger("my.test.logger.repeat")
        log = get_logger("my.test.logger.repeat")
        assert len(log.handlers) == 1

    def test_handler_uses_expected_format(self):
        """Test that the StreamHandler added by get_logger uses the expected log message format."""
        log = get_logger("my.test.logger.format")
        formatter = log.handlers[0].formatter
        assert formatter._fmt == "%(levelname)s (%(name)s): %(message)s"


class _FakeTqdm:
    """A fake tqdm class to simulate the behavior of tqdm for testing purposes."""
    instances = []

    def __init__(self, total=None, unit=None, unit_scale=None):
        self.total = total
        self.unit = unit
        self.unit_scale = unit_scale
        self.updates = []
        self.closed = False
        _FakeTqdm.instances.append(self)

    def update(self, n):
        self.updates.append(n)

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def _reset_progress_state(monkeypatch):
    """show_dl_progress tracks pbar/last_loaded as module-global mutable state - isolate tests from each other."""
    monkeypatch.setattr(logger_module, "tqdm", _FakeTqdm)
    monkeypatch.setattr(logger_module, "pbar", None)
    monkeypatch.setattr(logger_module, "last_loaded", 0)
    _FakeTqdm.instances = []
    yield


class TestShowDlProgress:
    """Tests for the show_dl_progress function, which updates a progress bar during downloads."""

    def test_creates_progress_bar_on_first_call(self):
        """Test that show_dl_progress creates a new progress bar on the first call."""
        show_dl_progress(block_num=0, block_size=100, total_size=1000)
        assert len(_FakeTqdm.instances) == 1
        assert _FakeTqdm.instances[0].total == 1000

    def test_updates_by_the_incremental_amount_downloaded(self):
        """Test that show_dl_progress updates the progress bar by the incremental amount downloaded."""
        show_dl_progress(block_num=1, block_size=100, total_size=1000)
        show_dl_progress(block_num=2, block_size=100, total_size=1000)
        bar = _FakeTqdm.instances[0]
        assert bar.updates == [100, 100]

    def test_reuses_the_same_bar_across_calls(self):
        """Test that show_dl_progress reuses the same progress bar instance across multiple calls."""
        show_dl_progress(block_num=1, block_size=100, total_size=1000)
        show_dl_progress(block_num=2, block_size=100, total_size=1000)
        assert len(_FakeTqdm.instances) == 1

    def test_closes_and_resets_state_when_download_completes(self):
        """Test that show_dl_progress closes the progress bar and resets state when the download completes."""
        show_dl_progress(block_num=1, block_size=1000, total_size=1000)
        bar = _FakeTqdm.instances[0]
        assert bar.closed is True
        assert logger_module.pbar is None
        assert logger_module.last_loaded == 0

    def test_starts_a_fresh_bar_after_a_completed_download(self):
        """Test that show_dl_progress starts a fresh progress bar after a completed download."""
        show_dl_progress(block_num=1, block_size=1000, total_size=1000)  # completes and resets
        show_dl_progress(block_num=0, block_size=100, total_size=500)  # a new download starts
        assert len(_FakeTqdm.instances) == 2
        assert _FakeTqdm.instances[1].total == 500
