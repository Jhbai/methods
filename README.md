---

# DBManager for SQLite

這是一個簡單的 Python 類別 `DBManager`，用於封裝 `sqlite3` 的常用操作，讓您能夠更輕鬆地管理 SQLite 資料庫。它支援上下文管理器 (`with` 語法)，並提供了一系列常用方法，如建立資料表、插入、更新、搜尋和刪除資料。

## ✨ 特性

- **輕鬆連線**：簡化了與 SQLite 資料庫的連線和關閉流程。
- **上下文管理器**：支援 `with` 語法，可自動提交和關閉連線，確保資源被正確釋放。
- **安全的參數化查詢**：所有插入和更新操作均使用參數化查詢 (`?`)，有效防止 SQL 注入攻擊。
- **物件導向**：將資料庫操作封裝在一個類別中，使程式碼更具結構性且易於重用。
- **常用的 CRUD 操作**：提供簡單易用的方法來執行建立資料表、插入、更新、刪除和搜尋等基本資料庫操作。
- **錯誤處理**：包含了基本的 `try-except` 區塊，用於捕捉和顯示 SQLite 操作中可能發生的錯誤。

## 🚀 如何使用

以下是如何使用 `DBManager` 類別的範例。

### 1. 建立資料庫連線並建立資料表

使用 `with` 語法可以確保連線在使用後自動關閉。

```python
from db_manager import DBManager # 假設您將程式碼儲存為 db_manager.py

db_name = "my_database.db"
table_name = "users"
columns = ["id", "name", "email"]
types = ["INTEGER PRIMARY KEY AUTOINCREMENT", "TEXT NOT NULL", "TEXT"]

with DBManager(db_name) as db:
    # 建立一個名為 'users' 的資料表
    if db.create_table(table_name, columns, types):
        print(f"資料表 '{table_name}' 建立成功！")

```

### 2. 插入資料

```python
with DBManager(db_name) as db:
    # 插入單筆資料
    user_columns = ["name", "email"]
    user_data = ["Alice", "alice@example.com"]
    if db.insert_table(table_name, user_columns, user_data):
        print("資料插入成功！")

    # 插入另一筆資料
    user_data_2 = ["Bob", "bob@example.com"]
    db.insert_table(table_name, user_columns, user_data_2)
```

### 3. 搜尋資料

`search` 方法接受一個完整的 SQL 查詢語句。

```python
with DBManager(db_name) as db:
    # 搜尋所有使用者
    all_users = db.search(f"SELECT * FROM {table_name}")
    print("所有使用者:", all_users)

    # 搜尋特定使用者
    specific_user = db.search(f"SELECT * FROM {table_name} WHERE name = 'Alice'")
    print("特定使用者 (Alice):", specific_user)
```

### 4. 更新資料

```python
with DBManager(db_name) as db:
    # 更新 Bob 的 email
    uid_column = "name"
    uid_value = "Bob"
    columns_to_update = ["email"]
    new_data = ["bob_new@example.com"]

    if db.update_table(table_name, uid_column, uid_value, columns_to_update, new_data):
        print("資料更新成功！")
    
    # 驗證更新結果
    updated_user = db.search(f"SELECT * FROM {table_name} WHERE name = 'Bob'")
    print("更新後的 Bob:", updated_user)
```

### 5. 清空資料表 (Truncate)

`truncate_table` 會刪除資料表中的所有資料，但保留資料表結構。

```python
with DBManager(db_name) as db:
    db.truncate_table(table_name)
```

### 6. 刪除資料表 (Drop)

`delete_table` 會完全移除整個資料表。

```python
with DBManager(db_name) as db:
    if db.delete_table(table_name):
        print(f"資料表 '{table_name}' 已成功刪除。")
```

## 📖 API 參考

### `__init__(self, db_name)`
- **描述**: 初始化 `DBManager` 物件並連線到指定的 SQLite 資料庫。
- **參數**:
  - `db_name` (str): 資料庫檔案的名稱。

### `create_table(self, table_name: str, columns: List[str], types: List[str]) -> bool`
- **描述**: 建立一個新的資料表。
- **參數**:
  - `table_name` (str): 要建立的資料表名稱。
  - `columns` (List[str]): 欄位名稱列表。
  - `types` (List[str]): 對應的欄位類型和約束列表 (例如 `["INTEGER PRIMARY KEY", "TEXT NOT NULL"]`)。
- **回傳**: `True` 表示成功，`False` 表示失敗。

### `insert_table(self, table_name: str, columns: List[str], data: List) -> bool`
- **描述**: 將一筆資料插入指定的資料表。
- **參數**:
  - `table_name` (str): 目標資料表名稱。
  - `columns` (List[str]): 要插入資料的欄位名稱列表。
  - `data` (List): 與 `columns` 對應的資料值列表。
- **回傳**: `True` 表示成功，`False` 表示失敗。

### `update_table(self, table_name: str, uid_column: str, uid: Any, columns: List[str], data: List) -> bool`
- **描述**: 更新符合條件的資料。
- **參數**:
  - `table_name` (str): 目標資料表名稱。
  - `uid_column` (str): 用於 `WHERE` 條件的欄位名稱 (例如 `id`)。
  - `uid` (Any): `WHERE` 條件中 `uid_column` 對應的值。
  - `columns` (List[str]): 要更新的欄位名稱列表。
  - `data` (List): 與 `columns` 對應的更新資料值列表。
- **回傳**: `True` 表示成功更新至少一筆資料，`False` 表示沒有找到匹配的資料或更新失敗。

### `search(self, sql: str) -> List[tuple]`
- **描述**: 執行一個 SQL 查詢語句並返回所有結果。
- **參數**:
  - `sql` (str): 要執行的完整 SQL `SELECT` 語句。
- **回傳**: 一個包含結果元組 (tuple) 的列表。如果沒有結果，則返回空列表。

### `truncate_table(self, table_name: str) -> bool`
- **描述**: 刪除一個資料表中的所有資料，但保留資料表結構。
- **參數**:
  - `table_name` (str): 要清空的資料表名稱。
- **回傳**: `True` 表示成功，`False` 表示失敗。

### `delete_table(self, table_name: str) -> bool`
- **描述**: 從資料庫中完全刪除一個資料表。
- **參數**:
  - `table_name` (str): 要刪除的資料表名稱。
- **回傳**: `True` 表示成功，`False` 表示失敗。

### `close(self)`
- **描述**: 提交任何待處理的交易並關閉資料庫連線。當使用 `with` 語法時，這個方法會被自動呼叫。
