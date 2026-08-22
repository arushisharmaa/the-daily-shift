import os
import json
import re
import datetime
import feedparser
import resend
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# Load environment variables from .env if running locally
load_dotenv()

# ---------------------------------------------------------------------------
# 1. FEEDS REGISTRY
# ---------------------------------------------------------------------------
CATEGORY_FEEDS = {
    "Big Tech & Startups": [
        "https://news.ycombinator.com/rss",
        "https://techcrunch.com/feed/",
        "https://www.theverge.com/rss/index.xml"
    ],
    "Artificial Intelligence": [
        "https://openai.com/news/rss.xml",
        "https://deepmind.google/blog/feed/basic/",
        "https://huggingface.co/blog/feed.xml"
    ],
    "Pop Culture": [
        "https://www.rollingstone.com/culture/culture-news/feed/",
        "https://variety.com/feed/"
    ],
    "World News": [
        "http://feeds.bbci.co.uk/news/world/rss.xml",
        "https://rss.nytimes.com/services/xml/rss/nyt/World.xml"
    ]
}

# ---------------------------------------------------------------------------
# 2. PYDANTIC SCHEMAS FOR STRUCTURED JSON
# ---------------------------------------------------------------------------
class NewsItem(BaseModel):
    title: str = Field(description="Bold headline summary, max 8 words.")
    body: str = Field(description="Single-sentence punchy summary explaining what happened.")
    link: str = Field(description="Direct URL to the original source article.")

class CategoryDigest(BaseModel):
    category_name: str = Field(description="The category name (e.g., Big Tech & Startups)")
    items: list[NewsItem] = Field(description="Exactly 2 to 3 high-impact news items.")

class DailyDigestPayload(BaseModel):
    date: str = Field(description="Date string in YYYY-MM-DD format.")
    categories: list[CategoryDigest]

# ---------------------------------------------------------------------------
# 3. RSS INGESTION & CLEANING
# ---------------------------------------------------------------------------
def clean_html(raw_html: str) -> str:
    """Strip HTML tags and excess whitespace."""
    clean_text = re.sub(r'<[^>]+>', '', raw_html)
    return " ".join(clean_text.split())[:300]

def fetch_raw_news() -> dict[str, list[dict]]:
    """Fetch raw entries from configured feeds."""
    raw_news = {}
    for category, urls in CATEGORY_FEEDS.items():
        raw_news[category] = []
        for url in urls:
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:3]:
                    summary = entry.get("summary", entry.get("description", ""))
                    raw_news[category].append({
                        "title": entry.title,
                        "link": entry.link,
                        "summary": clean_html(summary)
                    })
            except Exception as e:
                print(f"Error fetching feed {url}: {e}")
    return raw_news

# ---------------------------------------------------------------------------
# 4. LLM EXTRACTION (GEMINI)
# ---------------------------------------------------------------------------
def generate_digest_data(raw_data: dict[str, list[dict]]) -> DailyDigestPayload:
    """Send feed content to Gemini Flash and obtain structured payload."""
    client = genai.Client()  # Uses GEMINI_API_KEY from environment
    
    prompt_content = "RAW NEWS FEEDS TO PROCESS:\n\n"
    for category, articles in raw_data.items():
        prompt_content += f"=== CATEGORY: {category} ===\n"
        for art in articles:
            prompt_content += f"- Title: {art['title']}\n  Link: {art['link']}\n  Summary: {art['summary']}\n\n"

    system_instruction = """
    You are the head editor for 'The Daily Shift', a zero-fluff daily news briefing.
    Analyze the raw news stories and extract EXACTLY 2 to 3 top news items per category.
    
    RULES:
    1. Select only high-signal, major news stories. Ignore duplicate stories or routine PR fluff.
    2. 'title' must be a sharp, punchy headline (max 8 words).
    3. 'body' must be exactly ONE concise, factual sentence explaining what shifted or happened.
    4. Retain the exact original 'link' URL provided in the input source.
    """

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt_content,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=DailyDigestPayload,
            temperature=0.2,
        ),
    )
    
    return DailyDigestPayload.model_validate_json(response.text)

# ---------------------------------------------------------------------------
# 5. EMAIL DELIVERY (RESEND)
# ---------------------------------------------------------------------------
def send_email_digest(digest: DailyDigestPayload):
    """Build HTML template and dispatch email via Resend SDK."""
    resend_key = os.getenv("RESEND_API_KEY")
    recipient = os.getenv("RECIPIENT_EMAIL")

    if not resend_key or not recipient:
        raise ValueError("RESEND_API_KEY or RECIPIENT_EMAIL environment variable is missing.")

    resend.api_key = resend_key
    date_str = datetime.datetime.now().strftime("%B %d, %Y")

    sections_html = ""
    for cat in digest.categories:
        sections_html += f"""
        <div style="margin-top: 24px; margin-bottom: 12px; border-bottom: 2px solid #3b82f6; padding-bottom: 4px;">
            <h2 style="color: #60a5fa; font-size: 13px; letter-spacing: 1.2px; text-transform: uppercase; margin: 0;">{cat.category_name}</h2>
        </div>
        """
        for item in cat.items:
            sections_html += f"""
            <div style="margin-bottom: 14px; padding: 14px; background-color: #1e293b; border-radius: 8px;">
                <a href="{item.link}" style="color: #38bdf8; font-size: 15px; font-weight: bold; text-decoration: none;">{item.title}</a>
                <p style="color: #cbd5e1; font-size: 13px; margin: 6px 0 0 0; line-height: 1.4;">{item.body}</p>
            </div>
            """

    full_html = f"""
    <!DOCTYPE html>
    <html>
    <body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background-color: #0f172a; color: #f8fafc; padding: 24px; max-width: 600px; margin: 0 auto;">
      <div style="margin-bottom: 20px; border-bottom: 1px solid #334155; padding-bottom: 12px;">
        <span style="color: #34d399; font-size: 11px; font-weight: bold; letter-spacing: 1.5px;">THE DAILY SHIFT • {date_str.upper()}</span>
        <h1 style="color: #ffffff; font-size: 22px; margin: 4px 0 0 0;">What Moved Today</h1>
      </div>
      {sections_html}
      <div style="margin-top: 32px; padding-top: 16px; border-top: 1px solid #334155; font-size: 12px; color: #64748b; text-align: center;">
        The Daily Shift • Zero Fluff News
      </div>
    </body>
    </html>
    """

    params = {
        "from": "The Daily Shift <onboarding@resend.dev>",
        "to": [recipient],
        "subject": f"The Daily Shift - {date_str}",
        "html": full_html,
    }

    email_response = resend.Emails.send(params)
    print(f"Email sent successfully! Response ID: {email_response['id']}")

# ---------------------------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("1. Ingesting RSS feeds...")
    raw_news = fetch_raw_news()
    
    print("2. Generating 'The Daily Shift' 2-3 item summaries via Gemini...")
    digest_data = generate_digest_data(raw_news)
    
    print("3. Delivering daily email via Resend...")
    send_email_digest(digest_data)
    
    print("Done!")