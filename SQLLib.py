import sqlite3
from typing import List, Dict, Any

class DBManager:
    def __init__(self, db_name):
        try:
            self.conn = sqlite3.connect(db_name)
            self.cursor = self.conn.cursor()
            print(f"Success connect to myDB: {db_name}")
        except sqlite3.Error as e:
            print(f"DB Connect Error, error is: {e}")
            self.conn = None
            self.cursor = None

    def __enter__(self):
        """To support 'with' description for entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """To support 'with' description to close connection automatically"""
        self.close()

    def __del__(self):
        """If this object is deleted, then connection will be closed"""
        if self.conn:
            self.conn.close()

    def close(self):
        """Close connection"""
        if self.conn:
            self.conn.commit()
            self.conn.close()
    
    def create_table(self, table_name: str, columns: List[str], types: List[str]) -> bool:
        if not self.cursor:
            return False
        
        if len(columns) != len(types):
            print("Error: Columns amount is not equal to Types amount.")
            return False
        
        column_defs = ", ".join([f"{col} {typ}" for col, typ in zip(columns, types)])
        query = f"CREATE TABLE IF NOT EXISTS {table_name} ({column_defs})"

        try:
            self.cursor.execute(query)
            self.conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"Create Table '{table_name}' Failed: {e}")
            return False

    def update_table(self, table_name: str, uid_column: List[str], uid: str, columns: List[str], data: List) -> bool:
        if not self.cursor:
            return False
        if len(columns) != len(data):
            print("Error: Columns amount is not equal to Types amount.")
            return False
        
        set_clause = ", ".join([f"{col} = ?" for col in columns])
        query = f"UPDATE {table_name} SET {set_clause} WHERE {uid_column} = ?"

        try:
            params = tuple(data) + (uid,)
            self.cursor.execute(query, params)
            self.conn.commit()
            if self.cursor.rowcount > 0:
                return True
            else:
                print(f"In '{table_name}' , cannot find {uid_column} = {uid} to updated !")
                return False
        except sqlite3.Error as e:
            print(f"Update Table '{table_name}' Failed: {e}")
            return False
        
    def insert_table(self, table_name: str, columns: List[str], data: str) -> bool:
        if not self.cursor:
            return False
        if len(columns) != len(data):
            print("Error: Columns amount is not equal to Types amount.")
            return False
        placeholders = ", ".join(["?"] * len(columns))
        cols = ", ".join(columns)
        query = f"INSERT INTO {table_name} ({cols}) VALUES ({placeholders})"

        try:
            self.cursor.execute(query, tuple(data))
            self.conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"Insert Table '{table_name}' Failed: {e}")
            return False
        
    def delete_table(self, table_name: str) -> bool:
        if not self.cursor:
            return False

        query = f"DROP TABLE IF EXISTS {table_name}"
        try:
            self.cursor.execute(query)
            self.conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"Delete Table '{table_name}' Failed: {e}")
            return False
        
    def truncate_table(self, table_name: str) -> bool:
        if not self.cursor:
            return False

        query = f"DELETE FROM {table_name}"
        try:
            self.cursor.execute(query)
            self.conn.commit()
            print(f"Table '{table_name}' is truncated successfully")
            return True
        except sqlite3.Error as e:
            print(f"truncate Table '{table_name}' Failed: {e}")
            return False
        
    def search(self, sql):
        if not self.cursor:
            return ""
        self.cursor.execute(sql)
        return self.cursor.fetchall()
