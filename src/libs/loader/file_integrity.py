"""
File integrity checking for incremental ingestion.

This module provides abstract interfaces and implementations for checking
file integrity using SHA256 hashes. It supports incremental ingestion by
tracking which files have been successfully processed.

The default implementation uses SQLite with WAL mode for concurrent writes.

Author: Modular RAG MCP Server Project
License: MIT
"""

import hashlib
import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.observability.logger import get_logger

logger = get_logger(__name__)


@dataclass
class IntegrityRecord:
    """
    Record of a file's processing status.

    Attributes:
        file_hash: SHA256 hash of the file content
        file_path: Original file path
        status: Processing status (success, failed, pending)
        processed_at: Timestamp when processing completed
        error_msg: Error message if status is 'failed'
    """

    file_hash: str
    file_path: str
    status: str
    processed_at: Optional[datetime] = None
    error_msg: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "file_hash": self.file_hash,
            "file_path": self.file_path,
            "status": self.status,
            "processed_at": self.processed_at.isoformat() if self.processed_at else None,
            "error_msg": self.error_msg,
        }


class FileIntegrityChecker(ABC):
    """
    Abstract interface for file integrity checking.

    Implementations of this interface provide methods to:
    - Compute SHA256 hashes of file contents
    - Check if a file should be skipped (already successfully processed)
    - Mark files as successfully processed or failed
    """

    @abstractmethod
    def compute_sha256(self, file_path: str) -> str:
        """
        Compute SHA256 hash of a file's content.

        Args:
            file_path: Path to the file

        Returns:
            SHA256 hash as hexadecimal string

        Raises:
            FileNotFoundError: If the file doesn't exist
            IOError: If the file cannot be read
        """
        pass

    @abstractmethod
    def should_skip(self, file_hash: str) -> bool:
        """
        Check if a file should be skipped during ingestion.

        A file should be skipped if it has been successfully processed
        before with the same hash.

        Args:
            file_hash: SHA256 hash of the file content

        Returns:
            True if the file should be skipped, False otherwise
        """
        pass

    @abstractmethod
    def mark_success(
        self, file_hash: str, file_path: str, metadata: Optional[dict] = None
    ) -> None:
        """
        Mark a file as successfully processed.

        Args:
            file_hash: SHA256 hash of the file content
            file_path: Original file path
            metadata: Optional additional metadata to store
        """
        pass

    @abstractmethod
    def mark_failed(self, file_hash: str, error_msg: str) -> None:
        """
        Mark a file processing as failed.

        Args:
            file_hash: SHA256 hash of the file content
            error_msg: Error message describing the failure
        """
        pass

    @abstractmethod
    def get_record(self, file_hash: str) -> Optional[IntegrityRecord]:
        """
        Get the processing record for a file.

        Args:
            file_hash: SHA256 hash of the file content

        Returns:
            IntegrityRecord if found, None otherwise
        """
        pass


class SQLiteIntegrityChecker(FileIntegrityChecker):
    """
    SQLite-based implementation of FileIntegrityChecker.

    Uses SQLite with WAL (Write-Ahead Logging) mode to support
    concurrent writes from multiple processes.

    Database schema:
        Table: ingestion_history
        Columns:
            - file_hash TEXT PRIMARY KEY
            - file_path TEXT NOT NULL
            - status TEXT NOT NULL
            - processed_at TIMESTAMP
            - error_msg TEXT
            - metadata TEXT (JSON)

    The database file is created at: data/db/ingestion_history.db

    Example:
        >>> checker = SQLiteIntegrityChecker(db_path="data/db/ingestion_history.db")
        >>> file_hash = checker.compute_sha256("/path/to/document.pdf")
        >>> if not checker.should_skip(file_hash):
        ...     # Process the file
        ...     checker.mark_success(file_hash, "/path/to/document.pdf")
    """

    # Default database path
    DEFAULT_DB_PATH = "data/db/ingestion_history.db"

    # SQL statements
    CREATE_TABLE_SQL = """
        CREATE TABLE IF NOT EXISTS ingestion_history (
            file_hash TEXT PRIMARY KEY,
            file_path TEXT NOT NULL,
            status TEXT NOT NULL,
            processed_at TIMESTAMP,
            error_msg TEXT,
            metadata TEXT
        )
    """

    CREATE_INDEX_SQL = """
        CREATE INDEX IF NOT EXISTS idx_status ON ingestion_history(status)
    """

    SELECT_SUCCESS_SQL = """
        SELECT status FROM ingestion_history WHERE file_hash = ? AND status = 'success'
    """

    INSERT_RECORD_SQL = """
        INSERT OR REPLACE INTO ingestion_history
        (file_hash, file_path, status, processed_at, error_msg, metadata)
        VALUES (?, ?, ?, ?, ?, ?)
    """

    SELECT_RECORD_SQL = """
        SELECT file_hash, file_path, status, processed_at, error_msg
        FROM ingestion_history WHERE file_hash = ?
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize SQLite integrity checker.

        Args:
            db_path: Path to the SQLite database file (defaults to data/db/ingestion_history.db)
        """
        self.db_path = db_path or self.DEFAULT_DB_PATH
        self._conn: Optional[sqlite3.Connection] = None

        # Ensure database directory exists
        db_file = Path(self.db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_db()

        logger.info(f"Initialized SQLiteIntegrityChecker: db_path='{self.db_path}'")

    def _init_db(self) -> None:
        """Initialize database schema and enable WAL mode."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Create table
        cursor.execute(self.CREATE_TABLE_SQL)

        # Create index for faster queries
        cursor.execute(self.CREATE_INDEX_SQL)

        # Enable WAL mode for better concurrency
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")

        conn.commit()
        logger.debug("Database schema initialized with WAL mode")

    def _get_connection(self) -> sqlite3.Connection:
        """
        Get or create SQLite connection.

        Returns:
            SQLite connection object
        """
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row  # Enable row_factory for dict-like access
        return self._conn

    def compute_sha256(self, file_path: str) -> str:
        """
        Compute SHA256 hash of a file's content.

        Reads the file in chunks to handle large files efficiently.

        Args:
            file_path: Path to the file

        Returns:
            SHA256 hash as hexadecimal string

        Raises:
            FileNotFoundError: If the file doesn't exist
            IOError: If the file cannot be read

        Example:
            >>> file_hash = checker.compute_sha256("/path/to/document.pdf")
            >>> print(file_hash)
            'a1b2c3d4e5f6...'
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not path.is_file():
            raise IOError(f"Path is not a file: {file_path}")

        sha256_hash = hashlib.sha256()

        """
          3.1 "rb" 模式 - 二进制读取

          # 文本模式（默认）
          open("file.txt", "r")      # 按字符读取，Windows 会转换换行符
        
          # 二进制模式
          open("file.txt", "rb")     # 按字节读取，原始数据
        
          为什么用二进制？
          - SHA256 是按字节计算的
          - 避免换行符转换影响哈希值
          - 支持任何文件类型（图片、PDF等）
        """
        try:
            with open(path, "rb") as f:
                # Read in chunks for memory efficiency
                for chunk in iter(lambda: f.read(8192), b""):
                    sha256_hash.update(chunk)

            file_hash = sha256_hash.hexdigest()
            logger.debug(f"Computed SHA256 for '{file_path}': {file_hash[:16]}...")
            return file_hash

        except IOError as e:
            logger.error(f"Failed to read file '{file_path}': {e}")
            raise

    def should_skip(self, file_hash: str) -> bool:
        """
        Check if a file should be skipped during ingestion.

        A file should be skipped if it has been successfully processed
        before with the same hash.

        Args:
            file_hash: SHA256 hash of the file content

        Returns:
            True if the file should be skipped, False otherwise

        Example:
            >>> if checker.should_skip(file_hash):
            ...     print("File already processed, skipping")
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(self.SELECT_SUCCESS_SQL, (file_hash,))
        result = cursor.fetchone()

        should_skip = result is not None

        if should_skip:
            logger.debug(f"File with hash '{file_hash[:16]}...' should be skipped (already processed)")
        else:
            logger.debug(f"File with hash '{file_hash[:16]}...' should be processed")

        return should_skip

    def mark_success(
        self, file_hash: str, file_path: str, metadata: Optional[dict] = None
    ) -> None:
        """
        Mark a file as successfully processed.

        Args:
            file_hash: SHA256 hash of the file content
            file_path: Original file path
            metadata: Optional additional metadata to store

        Example:
            >>> checker.mark_success(file_hash, "/path/to/document.pdf", {"pages": 10})
        """
        import json

        conn = self._get_connection()
        cursor = conn.cursor()

        # Convert metadata to JSON if provided
        metadata_json = json.dumps(metadata) if metadata else None

        cursor.execute(
            self.INSERT_RECORD_SQL,
            (
                file_hash,
                file_path,
                "success",
                datetime.utcnow(),
                None,
                metadata_json,
            ),
        )

        conn.commit()
        logger.info(
            f"Marked file as successfully processed: hash='{file_hash[:16]}...', path='{file_path}'"
        )

    def mark_failed(self, file_hash: str, error_msg: str) -> None:
        """
        Mark a file processing as failed.

        Args:
            file_hash: SHA256 hash of the file content
            error_msg: Error message describing the failure

        Example:
            >>> try:
            ...     process_file()
            ... except Exception as e:
            ...     checker.mark_failed(file_hash, str(e))
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            self.INSERT_RECORD_SQL,
            (file_hash, "", "failed", datetime.utcnow(), error_msg, None),
        )

        conn.commit()
        logger.warning(
            f"Marked file as failed: hash='{file_hash[:16]}...', error='{error_msg}'"
        )

    def get_record(self, file_hash: str) -> Optional[IntegrityRecord]:
        """
        Get the processing record for a file.

        Args:
            file_hash: SHA256 hash of the file content

        Returns:
            IntegrityRecord if found, None otherwise

        Example:
            >>> record = checker.get_record(file_hash)
            >>> if record:
            ...     print(f"Status: {record.status}")
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(self.SELECT_RECORD_SQL, (file_hash,))
        row = cursor.fetchone()

        if row is None:
            return None

        return IntegrityRecord(
            file_hash=row["file_hash"],
            file_path=row["file_path"],
            status=row["status"],
            processed_at=datetime.fromisoformat(row["processed_at"]) if row["processed_at"] else None,
            error_msg=row["error_msg"],
        )

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
            logger.debug("Closed database connection")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def __del__(self):
        """Cleanup when instance is destroyed."""
        self.close()
