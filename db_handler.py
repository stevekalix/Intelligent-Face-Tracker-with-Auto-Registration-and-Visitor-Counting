"""
Database Handler Module supporting both MySQL and SQLite for Intelligent Face Tracker.
Manages persistent storage for visitor profiles, 512-D biometric embeddings (BLOB),
and audit-ready entry/exit event history.
"""

import os
import sqlite3
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger("VisitorDB")

try:
    import pymysql
    import pymysql.cursors
    HAS_PYMYSQL = True
except ImportError:
    HAS_PYMYSQL = False


class DatabaseHandler:
    """Universal Database Handler supporting both MySQL and SQLite."""

    def __init__(
        self,
        db_type: str = "sqlite",
        db_path: str = "visitor_system.db",
        mysql_config: Optional[dict] = None
    ):
        """Initialize database connection.

        Args:
            db_type: 'mysql' or 'sqlite'.
            db_path: Path to SQLite file (if using SQLite).
            mysql_config: Dictionary containing host, port, user, password, database.
        """
        self.db_type = db_type.lower()
        self.db_path = db_path
        self.mysql_config = mysql_config or {
            "host": "localhost",
            "port": 3306,
            "user": "root",
            "password": "root123",
            "database": "visitor_system"
        }

        self.sqlite_conn = None
        self.mysql_conn = None

        if self.db_type == "mysql":
            if not HAS_PYMYSQL:
                logger.warning("PyMySQL is not installed. Falling back to SQLite.")
                self.db_type = "sqlite"
                self._init_sqlite()
            else:
                self._init_mysql()
        else:
            self._init_sqlite()

    def _init_sqlite(self) -> None:
        """Initialize SQLite database connection and verify table schemas."""
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        self.sqlite_conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.sqlite_conn.row_factory = sqlite3.Row
        with self.sqlite_conn:
            cursor = self.sqlite_conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS visitors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id TEXT UNIQUE NOT NULL,
                    embedding BLOB NOT NULL,
                    first_seen TEXT NOT NULL,
                    last_seen TEXT NOT NULL,
                    total_appearances INTEGER DEFAULT 1,
                    sample_image_path TEXT
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id TEXT NOT NULL,
                    event_type TEXT NOT NULL CHECK(event_type IN ('ENTRY', 'EXIT', 'RECOGNIZED')),
                    timestamp TEXT NOT NULL,
                    confidence REAL DEFAULT 1.0,
                    image_path TEXT,
                    notes TEXT,
                    FOREIGN KEY (person_id) REFERENCES visitors(person_id)
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_visitors_person_id ON visitors(person_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_person_id ON events(person_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp)")
        logger.info("Connected to SQLite database at: %s", self.db_path)

    def _init_mysql(self) -> None:
        """Initialize MySQL connection and ensure database and tables exist."""
        cfg = self.mysql_config
        host = cfg.get("host", "localhost")
        port = int(cfg.get("port", 3306))
        user = cfg.get("user", "root")
        password = cfg.get("password", "root123")
        dbname = cfg.get("database", "visitor_system")

        try:
            # 1. Connect without db to ensure database exists
            root_conn = pymysql.connect(
                host=host,
                port=port,
                user=user,
                password=password,
                autocommit=True
            )
            with root_conn.cursor() as cur:
                cur.execute(f"CREATE DATABASE IF NOT EXISTS `{dbname}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
            root_conn.close()

            # 2. Connect to the target database
            self.mysql_conn = pymysql.connect(
                host=host,
                port=port,
                user=user,
                password=password,
                database=dbname,
                autocommit=True,
                cursorclass=pymysql.cursors.DictCursor
            )

            with self.mysql_conn.cursor() as cur:
                # Visitors table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS visitors (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        person_id VARCHAR(64) UNIQUE NOT NULL,
                        embedding LONGBLOB NOT NULL,
                        first_seen VARCHAR(32) NOT NULL,
                        last_seen VARCHAR(32) NOT NULL,
                        total_appearances INT DEFAULT 1,
                        sample_image_path VARCHAR(255),
                        INDEX idx_person_id (person_id)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                """)

                # Events table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS events (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        person_id VARCHAR(64) NOT NULL,
                        event_type VARCHAR(20) NOT NULL,
                        timestamp VARCHAR(32) NOT NULL,
                        confidence FLOAT DEFAULT 1.0,
                        image_path VARCHAR(255),
                        notes TEXT,
                        INDEX idx_person_id (person_id),
                        INDEX idx_timestamp (timestamp),
                        INDEX idx_type (event_type)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                """)

            logger.info("Connected to MySQL database [%s] on %s:%d as %s", dbname, host, port, user)
        except Exception as e:
            logger.error("Failed to connect to MySQL: %s. Falling back to SQLite.", e)
            self.db_type = "sqlite"
            self._init_sqlite()

    def get_next_visitor_id(self) -> str:
        """Generate the next sequential unique visitor ID (e.g., VISITOR_0001)."""
        if self.db_type == "mysql" and self.mysql_conn:
            with self.mysql_conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) AS total FROM visitors")
                row = cur.fetchone()
                count = row["total"] if row else 0
                return f"VISITOR_{count + 1:04d}"
        else:
            cursor = self.sqlite_conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM visitors")
            count = cursor.fetchone()[0]
            return f"VISITOR_{count + 1:04d}"

    def register_visitor(
        self,
        person_id: str,
        embedding: np.ndarray,
        image_path: Optional[str] = None,
        timestamp: Optional[str] = None
    ) -> bool:
        """Register a new unique visitor with 512-D binary embedding."""
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        embedding_blob = np.ascontiguousarray(embedding, dtype=np.float32).tobytes()

        try:
            if self.db_type == "mysql" and self.mysql_conn:
                with self.mysql_conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO visitors (person_id, embedding, first_seen, last_seen, total_appearances, sample_image_path)
                        VALUES (%s, %s, %s, %s, 1, %s)
                        """,
                        (person_id, embedding_blob, timestamp, timestamp, image_path),
                    )
            else:
                with self.sqlite_conn:
                    cursor = self.sqlite_conn.cursor()
                    cursor.execute(
                        """
                        INSERT INTO visitors (person_id, embedding, first_seen, last_seen, total_appearances, sample_image_path)
                        VALUES (?, ?, ?, ?, 1, ?)
                        """,
                        (person_id, embedding_blob, timestamp, timestamp, image_path),
                    )
            logger.info("Successfully registered new visitor [%s] in %s database.", person_id, self.db_type.upper())
            return True
        except Exception as e:
            logger.error("Error registering visitor [%s]: %s", person_id, e)
            return False

    def get_all_embeddings(self) -> Dict[str, np.ndarray]:
        """Fetch all registered face embeddings from the database."""
        embeddings_dict = {}
        try:
            if self.db_type == "mysql" and self.mysql_conn:
                with self.mysql_conn.cursor() as cur:
                    cur.execute("SELECT person_id, embedding FROM visitors")
                    rows = cur.fetchall()
                    for row in rows:
                        p_id = row["person_id"]
                        blob = row["embedding"]
                        vec = np.frombuffer(blob, dtype=np.float32)
                        norm = np.linalg.norm(vec)
                        if norm > 1e-6:
                            vec = vec / norm
                        embeddings_dict[p_id] = vec
            else:
                cursor = self.sqlite_conn.cursor()
                cursor.execute("SELECT person_id, embedding FROM visitors")
                rows = cursor.fetchall()
                for row in rows:
                    p_id = row["person_id"]
                    blob = row["embedding"]
                    vec = np.frombuffer(blob, dtype=np.float32)
                    norm = np.linalg.norm(vec)
                    if norm > 1e-6:
                        vec = vec / norm
                    embeddings_dict[p_id] = vec
            logger.debug("Loaded %d registered embeddings from %s database.", len(embeddings_dict), self.db_type.upper())
        except Exception as e:
            logger.error("Failed to load embeddings from database: %s", e)

        return embeddings_dict

    def update_visitor_seen(self, person_id: str, timestamp: Optional[str] = None) -> None:
        """Update the last_seen timestamp and increment total appearances count."""
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        try:
            if self.db_type == "mysql" and self.mysql_conn:
                with self.mysql_conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE visitors
                        SET last_seen = %s, total_appearances = total_appearances + 1
                        WHERE person_id = %s
                        """,
                        (timestamp, person_id),
                    )
            else:
                with self.sqlite_conn:
                    cursor = self.sqlite_conn.cursor()
                    cursor.execute(
                        """
                        UPDATE visitors
                        SET last_seen = ?, total_appearances = total_appearances + 1
                        WHERE person_id = ?
                        """,
                        (timestamp, person_id),
                    )
        except Exception as e:
            logger.error("Error updating last_seen for [%s]: %s", person_id, e)

    def log_event(
        self,
        person_id: str,
        event_type: str,
        timestamp: Optional[str] = None,
        confidence: float = 1.0,
        image_path: Optional[str] = None,
        notes: Optional[str] = ""
    ) -> bool:
        """Log an entry, exit, or recognition event."""
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        try:
            if self.db_type == "mysql" and self.mysql_conn:
                with self.mysql_conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO events (person_id, event_type, timestamp, confidence, image_path, notes)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        """,
                        (person_id, event_type, timestamp, float(confidence), image_path, notes),
                    )
            else:
                with self.sqlite_conn:
                    cursor = self.sqlite_conn.cursor()
                    cursor.execute(
                        """
                        INSERT INTO events (person_id, event_type, timestamp, confidence, image_path, notes)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (person_id, event_type, timestamp, float(confidence), image_path, notes),
                    )
            logger.info("Logged [%s] for [%s] in %s database.", event_type, person_id, self.db_type.upper())
            return True
        except Exception as e:
            logger.error("Failed to log event for [%s]: %s", person_id, e)
            return False

    def get_unique_visitor_count(self) -> int:
        """Get total count of unique visitors in database."""
        try:
            if self.db_type == "mysql" and self.mysql_conn:
                with self.mysql_conn.cursor() as cur:
                    cur.execute("SELECT COUNT(DISTINCT person_id) AS total FROM visitors")
                    row = cur.fetchone()
                    return row["total"] if row else 0
            else:
                cursor = self.sqlite_conn.cursor()
                cursor.execute("SELECT COUNT(DISTINCT person_id) FROM visitors")
                return cursor.fetchone()[0]
        except Exception as e:
            logger.error("Failed to fetch unique visitor count: %s", e)
            return 0

    def close(self) -> None:
        """Cleanly close database connections."""
        try:
            if self.sqlite_conn:
                self.sqlite_conn.close()
            if self.mysql_conn:
                self.mysql_conn.close()
            logger.info("Database connection closed.")
        except Exception as e:
            logger.warning("Error closing database: %s", e)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
