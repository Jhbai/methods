# main.py
from fastapi import FastAPI
import inference
import asyncio

app = FastAPI(title="StreamSponse Modular", version="2.0.0")

app.include_router(inference.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, workers=8, reload=True)
