from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from routes.upload import router as upload_router
from routes.chat import router as chat_router 

app = FastAPI(title="University AI Assistant")

# Mount the assets directory to serve files
app.mount("/assets", StaticFiles(directory="assets"), name="assets")

# Include routers
app.include_router(upload_router)
app.include_router(chat_router)

@app.get("/")
def read_root():
    return {"message": "University AI Assistant is running"}