from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from llm import LLMUnavailable, ResumeAnalysis, analyze_resume

app = FastAPI(title="Resume Analyzer")


class ResumeIn(BaseModel):
    text: str = Field(min_length=20, max_length=20000)


@app.post("/analyze", response_model=ResumeAnalysis)
async def analyze(body: ResumeIn):
    try:
        return await analyze_resume(body.text)
    except LLMUnavailable:
        raise HTTPException(status_code=503, detail="AI service temporarily unavailable")