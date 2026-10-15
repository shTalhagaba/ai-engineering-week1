import asyncio
import os
from typing import Literal

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from pydantic import BaseModel, Field, ValidationError

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY is missing from .env")

MODELS = tuple(
    m.strip()
    for m in os.getenv("GEMINI_MODELS", "gemini-3.5-flash,gemini-3.5-flash-lite").split(",")
    if m.strip()
)

client = genai.Client(api_key=api_key)


class ResumeAnalysis(BaseModel):
    candidate_name: str | None
    skills: list[str] = Field(default_factory=list)
    experience_years: int | None
    seniority: Literal["Junior", "Mid", "Senior", "Unknown"]


class LLMUnavailable(Exception):
    pass


async def analyze_resume(resume: str, retries: int = 3) -> ResumeAnalysis:
    prompt = f"""Analyze this resume and extract structured information.
Only use facts stated in the resume. If something is missing, use null
(or an empty list for skills, "Unknown" for seniority). Do not guess.

Resume:
{resume}
"""
    last_error = None

    for model in MODELS:
        for attempt in range(retries):
            try:
                response = await client.aio.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=ResumeAnalysis,
                        temperature=0,
                    ),
                )
                # Validation: JSON galat ya schema se match na kare to yahin pakra jayega
                result = ResumeAnalysis.model_validate_json(response.text)

                # Grounding check: jo skill resume mein hai hi nahi, hata do
                text = resume.lower()
                result.skills = [s for s in result.skills if s.lower() in text]
                return result

            except (ValidationError, ValueError) as exc:  # AI ka output kharab tha
                last_error = exc
                print(f"{model}: invalid output, retry {attempt + 1}/{retries}")

            except errors.ServerError as exc:  # 503 wagera
                last_error = exc
                await asyncio.sleep(2**attempt)

            except errors.ClientError as exc:
                last_error = exc
                if exc.code == 429:
                    await asyncio.sleep(2**attempt)
                else:  # 404 wagera: agla model
                    print(f"{model}: skipped ({exc.code})")
                    break

    raise LLMUnavailable(f"All models failed. Last error: {last_error}")