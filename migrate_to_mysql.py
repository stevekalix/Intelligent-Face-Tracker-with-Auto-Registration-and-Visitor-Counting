"""
Migration Utility: Copies all visitors and events from SQLite to MySQL.
"""

import sqlite3
import pymysql


def migrate(sqlite_path="visitor_system.db", host="localhost", port=3306, user="root", password="root123", dbname="visitor_system"):
    print(f"[INFO] Connecting to SQLite: {sqlite_path}")
    sq_conn = sqlite3.connect(sqlite_path)
    sq_conn.row_factory = sqlite3.Row
    sq_cur = sq_conn.cursor()

    print(f"[INFO] Connecting to MySQL '{dbname}' on {host}:{port} as '{user}'...")
    # Ensure db exists
    root_conn = pymysql.connect(host=host, port=port, user=user, password=password, autocommit=True)
    with root_conn.cursor() as cur:
        cur.execute(f"CREATE DATABASE IF NOT EXISTS `{dbname}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
    root_conn.close()

    my_conn = pymysql.connect(host=host, port=port, user=user, password=password, database=dbname, autocommit=True)
    my_cur = my_conn.cursor()

    # Recreate tables in MySQL
    my_cur.execute("""
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

    my_cur.execute("""
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

    # Migrate Visitors
    sq_cur.execute("SELECT person_id, embedding, first_seen, last_seen, total_appearances, sample_image_path FROM visitors")
    visitors = sq_cur.fetchall()
    v_count = 0
    for v in visitors:
        try:
            my_cur.execute(
                """
                INSERT INTO visitors (person_id, embedding, first_seen, last_seen, total_appearances, sample_image_path)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE last_seen=VALUES(last_seen), total_appearances=VALUES(total_appearances)
                """,
                (v["person_id"], v["embedding"], v["first_seen"], v["last_seen"], v["total_appearances"], v["sample_image_path"])
            )
            v_count += 1
        except Exception as e:
            print("Error migrating visitor:", e)

    # Migrate Events
    sq_cur.execute("SELECT person_id, event_type, timestamp, confidence, image_path, notes FROM events")
    events = sq_cur.fetchall()
    e_count = 0
    for e in events:
        try:
            my_cur.execute(
                """
                INSERT INTO events (person_id, event_type, timestamp, confidence, image_path, notes)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (e["person_id"], e["event_type"], e["timestamp"], e["confidence"], e["image_path"], e["notes"])
            )
            e_count += 1
        except Exception as e:
            print("Error migrating event:", e)

    sq_conn.close()
    my_conn.close()

    print(f"\n[SUCCESS] Migration Complete!")
    print(f" -> Migrated {v_count} unique visitors to MySQL table 'visitors'.")
    print(f" -> Migrated {e_count} events to MySQL table 'events'.")


if __name__ == "__main__":
    migrate()
