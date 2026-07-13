from fastapi import FastAPI

app = FastAPI(title="CareerPilot-AI API")

@app.get("/")
def read_root():
    return {"message": "CareerPilot-AI backend is running"}
