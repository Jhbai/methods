# ChromaDB 本地知識庫專案

這是一個使用 ChromaDB 和 SentenceTransformer 來建立本地知識庫的 Python 專案。您可以透過這個專案來儲存、管理和檢索非結構化文本資料，並利用語意搜尋來找到最相關的資訊。

## 專案概述

本專案的核心是利用 ChromaDB 這個開源的向量資料庫來儲存文本的嵌入(Embeddings)。 嵌入是文本的數字表示，能夠捕捉其語意。 專案使用 SentenceTransformer 模型將文本轉換為嵌入，這個模型特別擅長於理解句子的語意。

透過將文本資料轉換成語意向量並儲存在 ChromaDB 中，我們可以進行高效的相似性搜尋，快速地從大量資料中找出與使用者查詢最相關的內容。

## 功能

*   **本地部署**: 所有資料皆儲存在本地，確保資料的私密性。
*   **使用本地模型**: 透過載入本地的 SentenceTransformer 模型 (`all-MiniLM-L6-v2`)，無需依賴外部 API 即可將文本轉換為向量。
*   **持久化儲存**: 使用 `chromadb.PersistentClient` 確保儲存的知識在程式重新啟動後依然存在。
*   **簡易的知識存取**:
    *   `add_memory`: 一個簡單的函式，可以將新的文本訊息（知識）加入到資料庫中。
    *   `get_memory`: 透過輸入查詢語句，此函式會從資料庫中檢索出語意上最相近的 N 筆資料。

## 安裝需求

在執行此專案前，請確保您已經安裝了所有必要的 Python 套件。

```bash
pip install chromadb sentence-transformers
```

此外，您需要預先下載 `all-MiniLM-L6-v2` SentenceTransformer 模型並放置在指定的路徑 (`./LLM/all-MiniLM-L6-v2`)。您可以從 [Hugging Face Model Hub](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) 下載。

## 使用方法

1.  **初始化**: 執行 Python 腳本後，程式會自動初始化 ChromaDB 客戶端，並在指定路徑 (`./chroma_db`) 建立或載入資料庫檔案。
2.  **建立 Collection**: 程式會建立一個名為 `my_local_model_collection` 的 Collection（類似於資料庫中的資料表）來儲存您的知識。
3.  **儲存知識**:
    ```python
    from your_script_name import add_memory

    add_memory("question", "今天天氣如何？")
    add_memory("statement", "我喜歡吃蘋果。")
    ```
4.  **檢索知識**:
    ```python
    from your_script_name import get_memory

    query = "你喜歡什麼水果？"
    results = get_memory(query)
    print(results)
    ```

## 程式碼說明

*   `import uuid`: 用於為每一筆儲存的資料產生一個獨一無二的 ID。
*   `import chromadb`: 匯入 ChromaDB 函式庫。
*   `from chromadb.utils import embedding_functions`: 用於匯入 ChromaDB 提供的嵌入函式工具。
*   **載入本地嵌入模型**:
    ```python
    PATH = "./LLM/all-MiniLM-L6-v2"
    local_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=PATH)
    ```
    此部分程式碼會從指定的本地路徑載入 SentenceTransformer 模型作為嵌入函式。
*   **初始化 ChromaDB 客戶端**:
    ```python
    client = chromadb.PersistentClient(path="./chroma_db")
    ```
    建立一個持久化的 ChromaDB 客戶端，資料會被儲存在 `./chroma_db` 資料夾中。
*   **建立或取得 Collection**:
    ```python
    collection = client.get_or_create_collection(name=collection_name, embedding_function=local_ef)
    ```
    `get_or_create_collection` 函式會檢查是否存在名為 `my_local_model_collection` 的 Collection，如果不存在則會建立一個新的。 並且指定使用我們先前載入的本地模型作為其嵌入函式。
*   **`add_memory` 函式**:
    這個函式接收 `_type` 和 `message` 作為參數，將 `message` 包裝成文件，並為其附加元資料(metadata)和一個唯一的 ID，最後將這些資訊存入 ChromaDB 的 Collection 中。
*   **`get_memory` 函式**:
    此函式接收一個查詢 `message`，並使用 `collection.query` 方法來進行語意搜尋。 它會回傳與查詢最相似的前 10 筆結果，並在主控台印出每一筆結果的元資料和距離（距離越小代表越相似）。
