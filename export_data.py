"""
Data Export and Reporting Utility.
Extracts all registered visitor profiles, biometric entry/exit logs,
and image paths from MySQL or SQLite and exports them to CSV / console summary.
"""

import os
import json
import sqlite3
import csv
from datetime import datetime

try:
    import pymysql
    import pymysql.cursors
    HAS_PYMYSQL = True
except ImportError:
    HAS_PYMYSQL = False


def export_all_data(config_path="config.json", output_dir="exports"):
    """Export visitors and event logs from MySQL or SQLite into formatted CSV files."""
    db_type = "sqlite"
    mysql_cfg = {}
    sqlite_path = "visitor_system.db"

    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            db_info = cfg.get("database", {})
            db_type = db_info.get("db_type", "sqlite").lower()
            mysql_cfg = db_info.get("mysql", {})
            sqlite_path = db_info.get("db_path", "visitor_system.db")

    os.makedirs(output_dir, exist_ok=True)
    visitors = []
    events = []
    db_source_desc = ""

    # Case A: MySQL
    if db_type == "mysql" and HAS_PYMYSQL:
        host = mysql_cfg.get("host", "localhost")
        port = int(mysql_cfg.get("port", 3306))
        user = mysql_cfg.get("user", "root")
        password = mysql_cfg.get("password", "root123")
        dbname = mysql_cfg.get("database", "visitor_system")
        db_source_desc = f"MySQL [{dbname}] on {host}:{port} (User: {user})"

        conn = pymysql.connect(
            host=host, port=port, user=user, password=password,
            database=dbname, cursorclass=pymysql.cursors.DictCursor
        )
        with conn.cursor() as cur:
            cur.execute("SELECT person_id, first_seen, last_seen, total_appearances, sample_image_path FROM visitors ORDER BY id ASC")
            visitors = cur.fetchall()
            cur.execute("SELECT id, person_id, event_type, timestamp, confidence, image_path, notes FROM events ORDER BY id ASC")
            events = cur.fetchall()
        conn.close()

    # Case B: SQLite
    else:
        db_source_desc = f"SQLite [{os.path.abspath(sqlite_path)}]"
        if not os.path.exists(sqlite_path):
            print(f"[ERROR] SQLite Database '{sqlite_path}' not found!")
            return
        conn = sqlite3.connect(sqlite_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT person_id, first_seen, last_seen, total_appearances, sample_image_path FROM visitors ORDER BY id ASC")
        visitors = cur.fetchall()
        cur.execute("SELECT id, person_id, event_type, timestamp, confidence, image_path, notes FROM events ORDER BY id ASC")
        events = cur.fetchall()
        conn.close()

    # 1. Export Registered Visitors
    visitors_csv = os.path.join(output_dir, "registered_visitors.csv")
    with open(visitors_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Person ID", "First Seen", "Last Seen", "Total Appearances", "Sample Image Path"])
        for v in visitors:
            writer.writerow([v["person_id"], v["first_seen"], v["last_seen"], v["total_appearances"], v["sample_image_path"]])

    # 2. Export Entry / Exit Event Logs
    events_csv = os.path.join(output_dir, "event_logs.csv")
    with open(events_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Event ID", "Person ID", "Event Type", "Timestamp", "Confidence Score", "Cropped Image Path", "Notes / Duration"])
        for e in events:
            writer.writerow([e["id"], e["person_id"], e["event_type"], e["timestamp"], f"{float(e['confidence']):.2f}", e["image_path"], e["notes"]])

    # Count cropped images on disk
    entries_root = "logs/entries"
    total_images = 0
    if os.path.exists(entries_root):
        for root, _, files in os.walk(entries_root):
            total_images += len([f for f in files if f.endswith(".jpg")])

    print("\n" + "=" * 70)
    print("                     VISITOR SYSTEM DATA SUMMARY")
    print("=" * 70)
    print(f" • Active Database Backend        : {db_source_desc}")
    print(f" • Total Unique Visitors Enrolled : {len(visitors)}")
    print(f" • Total Event Logs (Entry/Exit)  : {len(events)}")
    print(f" • Total Cropped Face Images Saved: {total_images}")
    print(f" • Text Audit Log Location        : {os.path.abspath('events.log')}")
    print(f" • Cropped Images Folder          : {os.path.abspath(entries_root)}")
    print("-" * 70)
    print(f" [SUCCESS] Exported CSVs to directory: {os.path.abspath(output_dir)}")
    print(f"   1. {os.path.abspath(visitors_csv)}")
    print(f"   2. {os.path.abspath(events_csv)}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    export_all_data()
