from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field
from openai import OpenAI
import os
import logging

# Basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# Load API key from environment (set via Codespaces Secrets or 1Password)
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Please set OPENAI_API_KEY in your environment (check 1Password or Codespaces Secrets).")
client = OpenAI(api_key=api_key)
model = "gpt-4o-mini-2024-07-18"

# Data models
class EventExtraction(BaseModel):
    description: str
    is_calendar_event: bool
    confidence_score: float

class EventDetails(BaseModel):
    name: str
    date: str  # ISO 8601 format
    duration_minutes: int
    participants: list[str]

class EventConfirmation(BaseModel):
    confirmation_message: str
    calendar_link: Optional[str] = None

# Functions
def extract_event_info(user_input: str) -> EventExtraction:
    today = datetime.now().strftime("%A, %B %d, %Y")
    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": f"Today is {today}. Is this a calendar event?"},
            {"role": "user", "content": user_input},
        ],
        response_format=EventExtraction,
    )
    return completion.choices[0].message.parsed

def parse_event_details(description: str) -> EventDetails:
    today = datetime.now().strftime("%A, %B %d, %Y")
    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": f"Today is {today}. Extract event details."},
            {"role": "user", "content": description},
        ],
        response_format=EventDetails,
    )
    return completion.choices[0].message.parsed

def generate_confirmation(event_details: EventDetails) -> EventConfirmation:
    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": "Generate a confirmation. Sign off with 'Susie'."},
            {"role": "user", "content": str(event_details.model_dump())},
        ],
        response_format=EventConfirmation,
    )
    return completion.choices[0].message.parsed

# Main function
def process_calendar_request(user_input: str) -> Optional[EventConfirmation]:
    logger.info("Processing request...")
    extraction = extract_event_info(user_input)
    if not extraction.is_calendar_event or extraction.confidence_score < 0.7:
        logger.info("Not a calendar event or low confidence.")
        return None
    details = parse_event_details(extraction.description)
    return generate_confirmation(details)

if __name__ == "__main__":
    user_input = "Let's schedule a 1h team meeting next Tuesday at 2pm with Alice and Bob to discuss the project roadmap."
    result = process_calendar_request(user_input)
    if result:
        print(f"Confirmation: {result.confirmation_message}")
        if result.calendar_link:
            print(f"Calendar Link: {result.calendar_link}")
    else:
        print("This doesn't appear to be a calendar event.")
