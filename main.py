
import asyncio
import json
import os

from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError("GEMINI_API_KEY is missing from .env")

client = genai.Client(api_key=api_key)


async def analyze_resume(resume: str) -> dict:
    interaction = await client.aio.interactions.create(
        model="gemini-3.8-flash",
        input=f"""
        Analyze this resume and return structured information.

        Resume:
        {resume}

        Extract the candidate's name, skills, years of experience,
        and seniority level. If information is missing, use null or an empty list.
        Return only a valid JSON object with keys candidate_name, skills,
        experience_years, and seniority.
        """,
    )

    return json.loads(interaction.output_text)


async def main():
    resume = """
    Ali Khan is a software developer with 5 years of experience.
    He has worked with React Native, React, Node.js, TypeScript,
    and PostgreSQL.
    """

    result = await analyze_resume(resume)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
