if __name__ == '__main__':
    # --- 使用範例 ---
    db_manager = DBManager("example.db")

    # (1) 建立一個名為 'Users' 的資料表
    table_name = "Users"
    columns = ["id INTEGER PRIMARY KEY", "name TEXT NOT NULL", "email TEXT UNIQUE"]
    types = ["", "", ""]
    user_columns = ["id", "name", "email", "age"]
    user_types = ["INTEGER PRIMARY KEY AUTOINCREMENT", "TEXT NOT NULL", "TEXT", "INTEGER"]
    db_manager.create_table(table_name, user_columns, user_types)

    print("\n" + "="*20 + "\n")

    # (2) 插入幾筆資料
    db_manager.insert_table(table_name, ["name", "email", "age"], ["Alice", "alice@example.com", 30])
    db_manager.insert_table(table_name, ["name", "email", "age"], ["Bob", "bob@example.com", 25])
    db_manager.insert_table(table_name, ["name", "email", "age"], ["Charlie", "charlie@example.com", 35])
    print(db_manager.search("SELECT * FROM Users"))


    print("\n" + "="*20 + "\n")

    # (3) 更新資料 (更新 id=2 的資料)
    db_manager.update_table(table_name, "id", 2, ["email", "age"], ["bobby@newdomain.com", 28])
    # 更新一個不存在的 uid
    db_manager.update_table(table_name, "id", 99, ["name"], ["ghost"])
    print(db_manager.search("SELECT * FROM Users"))



    print("\n" + "="*20 + "\n")

    # (4) 清空 Table
    print(f"準備清空資料表 '{table_name}'...")
    db_manager.truncate_table(table_name)
    print(db_manager.search("SELECT * FROM Users"))



    print("\n" + "="*20 + "\n")

    # 重新插入一筆資料以供刪除表示範
    db_manager.insert_table(table_name, ["name", "email"], ["TestUser", "test@example.com"])
    print(db_manager.search("SELECT * FROM Users"))



    print("\n" + "="*20 + "\n")

    # (5) 刪除整個 Table
    print(f"準備刪除資料表 '{table_name}'...")
    db_manager.delete_table(table_name)
