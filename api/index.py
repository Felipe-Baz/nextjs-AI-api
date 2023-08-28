from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import nltk

from api.routers import sentiment

# Core Application Instance
app = FastAPI()

origins = [
    "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    nltk.download("vader_lexicon")


# Add Routers
app.include_router(sentiment.router, prefix="/api/v1", tags=["Analyze Feeling"])


# Health
@app.get("/api/v1/health", tags=["health"])
async def root():
    return {"version": "1.0.0"}
