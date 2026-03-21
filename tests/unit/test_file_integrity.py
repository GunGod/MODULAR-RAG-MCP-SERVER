"""
Unit tests for FileIntegrityChecker.

These tests verify the file integrity checking functionality including
SHA256 hash computation, skip logic, and database operations.

Author: Modular RAG MCP Server Project
License: MIT
"""

import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.libs.loader.file_integrity import (
    FileIntegrityChecker,
    IntegrityRecord,
    SQLiteIntegrityChecker,
)


class TestIntegrityRecord:
    """Test IntegrityRecord dataclass."""

    def test_create_record(self):
        """Test creating an IntegrityRecord."""
        record = IntegrityRecord(
            file_hash="abc123",
            file_path="/path/to/file.pdf",
            status="success"
        )

        assert record.file_hash == "abc123"
        assert record.file_path == "/path/to/file.pdf"
        assert record.status == "success"
        assert record.processed_at is None
        assert record.error_msg is None

    def test_to_dict(self):
        """Test IntegrityRecord.to_dict method."""
        from datetime import datetime

        processed_at = datetime(2024, 1, 1, 12, 0, 0)
        record = IntegrityRecord(
            file_hash="abc123",
            file_path="/path/to/file.pdf",
            status="success",
            processed_at=processed_at
        )

        result = record.to_dict()

        assert result["file_hash"] == "abc123"
        assert result["file_path"] == "/path/to/file.pdf"
        assert result["status"] == "success"
        assert result["processed_at"] == "2024-01-01T12:00:00"
        assert result["error_msg"] is None


class TestSQLiteIntegrityChecker:
    """Test SQLiteIntegrityChecker implementation."""

    def test_init_with_default_path(self):
        """Test initialization with default database path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            checker = SQLiteIntegrityChecker(db_path=str(db_path))

            assert checker.db_path == str(db_path)
            assert Path(db_path).exists()

            checker.close()

    def test_init_creates_directory(self):
        """Test that initialization creates database directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "subdir" / "test.db"
            checker = SQLiteIntegrityChecker(db_path=str(db_path))

            assert db_path.parent.exists()
            assert db_path.exists()

            checker.close()

    def test_init_creates_table(self):
        """Test that initialization creates the database table."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            SQLiteIntegrityChecker(db_path=str(db_path))

            # Verify table exists
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='ingestion_history'"
            )
            result = cursor.fetchone()

            assert result is not None
            assert result[0] == "ingestion_history"

            conn.close()

    def test_init_enables_wal_mode(self):
        """Test that initialization enables WAL mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            SQLiteIntegrityChecker(db_path=str(db_path))

            # Check WAL mode
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            cursor.execute("PRAGMA journal_mode")
            result = cursor.fetchone()

            assert result[0] == "wal"

            conn.close()

    def test_compute_sha256(self):
        """Test SHA256 hash computation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            checker = SQLiteIntegrityChecker(db_path=str(db_path))

            # Create a test file
            test_file = Path(tmpdir) / "test.txt"
            test_content = b"Hello, World!"
            test_file.write_bytes(test_content)

            # Compute hash
            file_hash = checker.compute_sha256(str(test_file))

            # Verify hash matches expected value
            import hashlib
            expected_hash = hashlib.sha256(test_content).hexdigest()
            assert file_hash == expected_hash
            assert len(file_hash) == 64  # SHA256 produces 64 hex characters

            checker.close()

    def test_compute_sha256_file_not_found(self):
        """Test that non-existent file raises FileNotFoundError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            checker = SQLiteIntegrityChecker(db_path=str(db_path))

            with pytest.raises(FileNotFoundError, match="File not found"):
                checker.compute_sha256("/nonexistent/file.txt")

            checker.close()

    def test_compute_sha256_consistent(self):
        """Test that hash computation is consistent for same file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            checker = SQLiteIntegrityChecker(db_path=str(db_path))

            # Create a test file
            test_file = Path(tmpdir) / "test.txt"
            test_content = b"Test content for hash consistency"
            test_file.write_bytes(test_content)

            # Compute hash multiple times
            hash1 = checker.compute_sha256(str(test_file))
            hash2 = checker.compute_sha256(str(test_file))
            hash3 = checker.compute_sha256(str(test_file))

            assert hash1 == hash2 == hash3

            checker.close()

    def test_should_skip_false_for_new_file(self):
        """Test should_skip returns False for new file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            checker = SQLiteIntegrityChecker(db_path=str(db_path))

            file_hash = "abc123def456"
            should_skip = checker.should_skip(file_hash)

            assert should_skip is False

            checker.close()

    def test_should_skip_true_after_mark_success(self):
        """Test should_skip returns True after marking success."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            checker = SQLiteIntegrityChecker(db_path=str(db_path))

            file_hash = "abc123def456"

            # Initially should not skip
            assert checker.should_skip(file_hash) is False

            # Mark as success
            checker.mark_success(file_hash, "/path/to/file.pdf")

            # Now should skip
            assert checker.should_skip(file_hash) is True

            checker.close()

    def test_should_skip_false_for_failed_file(self):
        """Test should_skip returns False for failed file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            checker = SQLiteIntegrityChecker(db_path=str(db_path))

            file_hash = "abc123def456"

            # Mark as failed
            checker.mark_failed(file_hash, "Processing error")

            # Should not skip failed files
            assert checker.should_skip(file_hash) is False

            checker.close()

    def test_mark_success(self):
        """Test marking a file as successfully processed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            checker = SQLiteIntegrityChecker(db_path=str(db_path))

            file_hash = "abc123"
            file_path = "/path/to/file.pdf"

            checker.mark_success(file_hash, file_path)

            # Verify record was created
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row  # Enable row_factory
            cursor = conn.cursor()

            cursor.execute("SELECT * FROM ingestion_history WHERE file_hash = ?", (file_hash,))
            row = cursor.fetchone()

            assert row is not None
            assert row["file_hash"] == file_hash
            assert row["file_path"] == file_path
            assert row["status"] == "success"
            assert row["error_msg"] is None

            conn.close()
            checker.close()

    def test_mark_success_with_metadata(self):
        """Test marking success with metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            checker = SQLiteIntegrityChecker(db_path=str(db_path))

            file_hash = "abc123"
            metadata = {"pages": 10, "title": "Test Document"}

            checker.mark_success(file_hash, "/path/to/file.pdf", metadata=metadata)

            # Verify metadata was stored
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row  # Enable row_factory
            cursor = conn.cursor()

            cursor.execute("SELECT metadata FROM ingestion_history WHERE file_hash = ?", (file_hash,))
            row = cursor.fetchone()

            assert row is not None
            stored_metadata = json.loads(row["metadata"])
            assert stored_metadata == metadata

            conn.close()
            checker.close()

    def test_mark_failed(self):
        """Test marking a file as failed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            checker = SQLiteIntegrityChecker(db_path=str(db_path))

            file_hash = "abc123"
            error_msg = "File corrupted"

            checker.mark_failed(file_hash, error_msg)

            # Verify record was created
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row  # Enable row_factory
            cursor = conn.cursor()

            cursor.execute("SELECT * FROM ingestion_history WHERE file_hash = ?", (file_hash,))
            row = cursor.fetchone()

            assert row is not None
            assert row["status"] == "failed"
            assert row["error_msg"] == error_msg

            conn.close()
            checker.close()

    def test_mark_failed_overrides_success(self):
        """Test that marking failed overrides previous success."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            checker = SQLiteIntegrityChecker(db_path=str(db_path))

            file_hash = "abc123"

            # Mark as success
            checker.mark_success(file_hash, "/path/to/file.pdf")
            assert checker.should_skip(file_hash) is True

            # Mark as failed
            checker.mark_failed(file_hash, "Processing failed")
            assert checker.should_skip(file_hash) is False

            checker.close()

    def test_mark_success_overrides_failed(self):
        """Test that marking success overrides previous failure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            checker = SQLiteIntegrityChecker(db_path=str(db_path))

            file_hash = "abc123"

            # Mark as failed
            checker.mark_failed(file_hash, "Processing failed")
            assert checker.should_skip(file_hash) is False

            # Mark as success
            checker.mark_success(file_hash, "/path/to/file.pdf")
            assert checker.should_skip(file_hash) is True

            checker.close()

    def test_get_record(self):
        """Test getting a processing record."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            checker = SQLiteIntegrityChecker(db_path=str(db_path))

            file_hash = "abc123"
            file_path = "/path/to/file.pdf"

            # Mark as success
            checker.mark_success(file_hash, file_path)

            # Get record
            record = checker.get_record(file_hash)

            assert record is not None
            assert isinstance(record, IntegrityRecord)
            assert record.file_hash == file_hash
            assert record.file_path == file_path
            assert record.status == "success"
            assert record.processed_at is not None

            checker.close()

    def test_get_record_not_found(self):
        """Test getting record for non-existent file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            checker = SQLiteIntegrityChecker(db_path=str(db_path))

            record = checker.get_record("nonexistent")

            assert record is None

            checker.close()

    def test_context_manager(self):
        """Test using checker as context manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            with SQLiteIntegrityChecker(db_path=str(db_path)) as checker:
                file_hash = "abc123"
                checker.mark_success(file_hash, "/path/to/file.pdf")

            # Connection should be closed after context
            # Verify data persists
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row  # Enable row_factory
            cursor = conn.cursor()

            cursor.execute("SELECT * FROM ingestion_history WHERE file_hash = ?", (file_hash,))
            row = cursor.fetchone()

            assert row is not None
            assert row["status"] == "success"

            conn.close()

    def test_default_db_path(self):
        """Test that default database path is correct."""
        import shutil

        # Save current directory
        original_cwd = Path.cwd()

        try:
            # Change to project root
            import os
            os.chdir(original_cwd.parent)  # Go up from tests/unit to project root

            # Remove existing data/db if it exists (for clean test)
            db_dir = Path("data/db")
            if db_dir.exists():
                shutil.rmtree(db_dir)

            # Create checker with default path
            checker = SQLiteIntegrityChecker()

            # Verify database file was created at default path
            expected_db_path = Path("data/db/ingestion_history.db")
            assert expected_db_path.exists(), f"Database file not created at {expected_db_path}"
            assert checker.db_path == SQLiteIntegrityChecker.DEFAULT_DB_PATH

            # Verify table exists
            import sqlite3
            conn = sqlite3.connect(str(expected_db_path))
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='ingestion_history'"
            )
            result = cursor.fetchone()
            assert result is not None
            conn.close()

            checker.close()

        finally:
            # Restore directory
            os.chdir(original_cwd)

            # Cleanup
            db_dir = Path("data/db")
            if db_dir.exists():
                shutil.rmtree(db_dir)

    def test_concurrent_write_simulation(self):
        """Test that WAL mode allows concurrent operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            # Create first checker
            checker1 = SQLiteIntegrityChecker(db_path=str(db_path))
            checker1.mark_success("hash1", "/path/to/file1.pdf")

            # Create second checker (simulates concurrent process)
            checker2 = SQLiteIntegrityChecker(db_path=str(db_path))
            checker2.mark_success("hash2", "/path/to/file2.pdf")

            # Both should succeed
            assert checker1.get_record("hash1") is not None
            assert checker2.get_record("hash2") is not None

            # Both should see each other's records
            assert checker1.get_record("hash2") is not None
            assert checker2.get_record("hash1") is not None

            checker1.close()
            checker2.close()

    def test_large_file_hashing(self):
        """Test hashing large files with chunked reading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            checker = SQLiteIntegrityChecker(db_path=str(db_path))

            # Create a file larger than buffer size (8192 bytes)
            test_file = Path(tmpdir) / "large.txt"
            large_content = b"x" * 100000  # 100KB file
            test_file.write_bytes(large_content)

            # Should handle large file without issues
            file_hash = checker.compute_sha256(str(test_file))

            assert file_hash is not None
            assert len(file_hash) == 64

            # Verify consistency
            file_hash2 = checker.compute_sha256(str(test_file))
            assert file_hash == file_hash2

            checker.close()


class TestFileIntegrityCheckerAbstract:
    """Test FileIntegrityChecker abstract interface."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that FileIntegrityChecker cannot be instantiated directly."""
        with pytest.raises(TypeError):
            FileIntegrityChecker()

    def test_concrete_implementation(self):
        """Test creating a concrete implementation."""
        class MockFileIntegrityChecker(FileIntegrityChecker):
            def __init__(self):
                self.records = {}

            def compute_sha256(self, file_path: str) -> str:
                return "mock_hash"

            def should_skip(self, file_hash: str) -> bool:
                return file_hash in self.records

            def mark_success(self, file_hash: str, file_path: str, metadata=None):
                self.records[file_hash] = {"status": "success", "path": file_path}

            def mark_failed(self, file_hash: str, error_msg: str):
                self.records[file_hash] = {"status": "failed", "error": error_msg}

            def get_record(self, file_hash: str):
                return self.records.get(file_hash)

        # Should be able to instantiate
        checker = MockFileIntegrityChecker()
        assert checker.compute_sha256("/path/to/file.pdf") == "mock_hash"
        assert checker.should_skip("new_hash") is False

        # Mark success
        checker.mark_success("hash1", "/path/to/file.pdf")
        assert checker.should_skip("hash1") is True
        assert checker.get_record("hash1") is not None
