#!/usr/bin/env python3
"""
Extract human people mentioned in CSV article text, including titles, frequencies,
and contextual sample text.

Built for a CSV where column B contains article text by default.

Outputs:
  1) people_mentions_long.csv
     One row per person-name mention, with sample_text context.

  2) people_title_mentions.csv
     Aggregated frequency by person_name + title, with sample_contexts.

  3) title_frequency.csv
     Aggregated frequency by title.

LLM mode is recommended when you need person-only extraction and want to avoid
organizations, locations, laws, facilities, and program names.

Install:
  pip install pandas openai python-dotenv tqdm

Set API key:
  export OPENAI_API_KEY="your_key_here"

Example, LLM extraction:
  python extract_people_titles_llm.py portsmouth_relevance_sample_50_articles_review.csv \
    --method llm \
    --model gpt-4.1-mini \
    --outdir people_title_outputs_llm

Example, deterministic fallback only:
  python extract_people_titles_llm.py portsmouth_relevance_sample_50_articles_review.csv \
    --method regex \
    --outdir people_title_outputs_regex
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


TITLE_WORDS = [
    "CEO", "CFO", "COO", "CTO", "CIO", "CRO", "CMO",
    "Chief Executive Officer", "Chief Financial Officer", "Chief Operating Officer",
    "Chief Technology Officer", "Chief Information Officer", "Chief Revenue Officer",
    "Chief Marketing Officer", "Chief", "Chairman", "Chairwoman", "Chair",
    "President", "Vice President", "VP", "Founder", "Co-founder", "Co Founder",
    "Director", "Executive Director", "Managing Director", "Senior Director",
    "Partner", "Principal", "Manager", "General Manager", "Spokesperson",
    "Secretary", "Deputy Secretary", "Assistant Secretary", "Administrator",
    "Commissioner", "Governor", "Mayor", "Senator", "Rep", "Representative",
    "Dr", "Dr.", "Professor", "Prof", "Prof.", "Mr", "Mr.", "Mrs", "Mrs.", "Ms", "Ms.",
]

TITLE_CORE = r"(?:" + "|".join(re.escape(t) for t in sorted(TITLE_WORDS, key=len, reverse=True)) + r")"
NAME_RE = re.compile(r"\b([A-Z][a-zA-Z.'-]+(?:\s+[A-Z][a-zA-Z.'-]+){1,3})\b")
TITLED_NAME_RE = re.compile(
    rf"(?P<title>(?:(?:[A-Z][A-Za-z&.'-]+|U\.S\.|US)\s+){{0,6}}{TITLE_CORE}"
    rf"(?:\s+(?:and|&)\s+{TITLE_CORE})?(?:\s+of\s+(?:(?:[A-Z][A-Za-z&.'-]+|U\.S\.|US)\s*){{1,5}})?)\s+"
    rf"(?P<name>[A-Z][a-zA-Z.'-]+(?:\s+[A-Z][a-zA-Z.'-]+){{1,3}})",
    flags=re.MULTILINE,
)

BAD_NAME_PARTS = {
    "Department", "Energy", "Commerce", "SoftBank", "Ohio", "Pike", "County",
    "Portsmouth", "Gaseous", "Diffusion", "Plant", "Technology", "Campus",
    "United", "States", "America", "Friday", "Monday", "Tuesday", "Wednesday",
    "Thursday", "Saturday", "Sunday", "January", "February", "March", "April",
    "May", "June", "July", "August", "September", "October", "November", "December",
    "Market", "Cap", "Enterprise", "Value", "Share", "Price", "Week", "High",
    "Act", "Law", "Bill", "Project", "Program", "Agency", "Office", "Council",
}


def normalize_space(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def normalize_person(name: str) -> str:
    name = normalize_space(name)
    name = re.sub(r"^(?:Dr\.?|Prof\.?|Mr\.?|Mrs\.?|Ms\.?)\s+", "", name)
    name = re.sub(r"'s$", "", name)
    return name.strip(" ,.;:()[]{}\"'")


def normalize_title(title: Optional[str]) -> str:
    return normalize_space(title).strip(" ,.;:()[]{}\"'") if title else ""


def likely_person_name(name: str) -> bool:
    parts = normalize_person(name).split()
    if len(parts) < 2 or len(parts) > 4:
        return False
    if any(p in BAD_NAME_PARTS for p in parts):
        return False
    if any(len(p) == 1 for p in parts):
        return False
    if any(p.isupper() and len(p) > 1 for p in parts):
        return False
    return True


def context_window(text: str, start: int, end: int, window: int = 180) -> str:
    left = max(0, start - window)
    right = min(len(text), end + window)
    snippet = normalize_space(text[left:right])
    if left > 0:
        snippet = "..." + snippet
    if right < len(text):
        snippet = snippet + "..."
    return snippet


def title_before_name(text: str, start_char: int) -> str:
    prefix = text[max(0, start_char - 140):start_char]
    prefix = re.split(r"[\n.!?;:]", prefix)[-1]
    prefix = normalize_space(prefix)
    m = re.search(
        rf"(?P<title>(?:(?:[A-Z][A-Za-z&.'-]+|U\.S\.|US)\s+){{0,6}}{TITLE_CORE}"
        rf"(?:\s+(?:and|&)\s+{TITLE_CORE})?(?:\s+of\s+(?:(?:[A-Z][A-Za-z&.'-]+|U\.S\.|US)\s*){{1,5}})?)\s*$",
        prefix,
    )
    return normalize_title(m.group("title")) if m else ""


def extract_mentions_regex(text: str) -> List[Dict[str, Any]]:
    text = str(text or "")
    mentions: List[Dict[str, Any]] = []
    seen = set()

    for m in TITLED_NAME_RE.finditer(text):
        name = normalize_person(m.group("name"))
        if not likely_person_name(name):
            continue
        span = (m.start("name"), m.end("name"))
        mentions.append({
            "person_name": name,
            "mention_text": m.group("name"),
            "title": normalize_title(m.group("title")),
            "sample_text": context_window(text, m.start(), m.end()),
            "start_char": m.start("name"),
            "end_char": m.end("name"),
            "match_method": "title_regex",
        })
        seen.add(span)

    for m in NAME_RE.finditer(text):
        name = normalize_person(m.group(1))
        span = (m.start(1), m.end(1))
        if span in seen or not likely_person_name(name):
            continue
        mentions.append({
            "person_name": name,
            "mention_text": m.group(1),
            "title": title_before_name(text, m.start(1)),
            "sample_text": context_window(text, m.start(1), m.end(1)),
            "start_char": m.start(1),
            "end_char": m.end(1),
            "match_method": "regex_person_candidate",
        })
        seen.add(span)
    return mentions


PERSON_SCHEMA = {
    "type": "object",
    "properties": {
        "mentions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "person_name": {
                        "type": "string",
                        "description": "Canonical full name of a human person. Do not return organizations, agencies, facilities, laws, projects, or places."
                    },
                    "mention_text": {
                        "type": "string",
                        "description": "The exact text used in the article mention, such as a full name, last name, or titled name."
                    },
                    "title": {
                        "type": ["string", "null"],
                        "description": "Professional, political, academic, or honorific title attached to this mention, such as CEO, Dr., U.S. Energy Secretary. Null if no title is stated nearby."
                    },
                    "sample_text": {
                        "type": "string",
                        "description": "A short exact or near-exact article excerpt surrounding the name that contextualizes the reference."
                    }
                },
                "required": ["person_name", "mention_text", "title", "sample_text"],
                "additionalProperties": False,
            }
        }
    },
    "required": ["mentions"],
    "additionalProperties": False,
}


def build_llm_prompt(article_text: str, max_chars: int) -> str:
    article = str(article_text or "")[:max_chars]
    return f"""
Extract all named references to HUMAN PEOPLE in the article below.

Rules:
- Return only real people, not organizations, agencies, facilities, laws, programs, projects, companies, locations, or publications.
- Include titles when stated near the name, for example: CEO, CRO, Dr., Professor, Senator, U.S. Energy Secretary, SoftBank Chairman and CEO.
- Count repeated named references. If a person is first named as "Masayoshi Son" and later referenced as "Son", include both mentions with person_name = "Masayoshi Son" when the article makes that clear.
- Do not include pronouns such as he, she, they.
- sample_text should be a short excerpt, ideally one sentence or sentence fragment, that shows the name/title in context.
- If no human people are named, return an empty mentions list.

ARTICLE:
{article}
""".strip()


def get_openai_client():
    try:
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError("The openai package is required for --method llm. Install with: pip install openai") from exc
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Export it before running --method llm.")
    return OpenAI()


def extract_mentions_llm(
    text: str,
    client: Any,
    model: str,
    max_chars: int,
    retries: int = 3,
    sleep_seconds: float = 1.5,
) -> List[Dict[str, Any]]:
    prompt = build_llm_prompt(text, max_chars=max_chars)
    last_error: Optional[Exception] = None

    for attempt in range(retries):
        try:
            response = client.responses.create(
                model=model,
                input=[
                    {
                        "role": "system",
                        "content": "You extract structured data from news articles. Be conservative and return only human people.",
                    },
                    {"role": "user", "content": prompt},
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "people_mentions",
                        "strict": True,
                        "schema": PERSON_SCHEMA,
                    }
                },
            )
            raw = response.output_text
            parsed = json.loads(raw)
            mentions = parsed.get("mentions", [])
            cleaned = []
            for item in mentions:
                name = normalize_person(item.get("person_name", ""))
                mention_text = normalize_space(item.get("mention_text", ""))
                sample_text = normalize_space(item.get("sample_text", ""))
                title = normalize_title(item.get("title") or "")
                if not name or not mention_text or not sample_text:
                    continue
                # Light guardrail: reject obvious organization-like names.
                if any(part in BAD_NAME_PARTS for part in name.split()):
                    continue
                cleaned.append({
                    "person_name": name,
                    "mention_text": mention_text,
                    "title": title,
                    "sample_text": sample_text,
                    "start_char": "",
                    "end_char": "",
                    "match_method": "openai_llm",
                })
            return cleaned
        except Exception as exc:  # pragma: no cover - network/API path
            last_error = exc
            if attempt < retries - 1:
                time.sleep(sleep_seconds * (attempt + 1))
            else:
                raise RuntimeError(f"OpenAI extraction failed after {retries} attempts: {last_error}")
    return []


def summarize(long_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if long_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    def first_contexts(series: pd.Series, limit: int = 5, max_chars: int = 1200) -> str:
        values = []
        seen = set()
        for value in series:
            text = normalize_space(value)
            if not text or text in seen:
                continue
            values.append(text)
            seen.add(text)
            if len(values) >= limit:
                break
        joined = " || ".join(values)
        return joined[:max_chars]

    people = (
        long_df.groupby(["person_name", "title"], dropna=False)
        .agg(
            mention_count=("person_name", "size"),
            article_count=("row_number", "nunique"),
            article_rows=("row_number", lambda s: ", ".join(map(str, sorted(set(s))))),
            sample_contexts=("sample_text", first_contexts),
            mention_text_examples=("mention_text", lambda s: ", ".join(list(dict.fromkeys(map(str, s)))[:10])),
            headlines=("headline", lambda s: " | ".join(dict.fromkeys([str(x) for x in s if pd.notna(x) and str(x).strip()]))[:500]),
        )
        .reset_index()
        .sort_values(["mention_count", "article_count", "person_name"], ascending=[False, False, True])
    )

    titled = long_df[long_df["title"].fillna("").str.len() > 0]
    titles = (
        titled.groupby("title", dropna=False)
        .agg(
            mention_count=("title", "size"),
            distinct_people=("person_name", "nunique"),
            people=("person_name", lambda s: ", ".join(sorted(set(map(str, s))))[:500]),
            sample_contexts=("sample_text", first_contexts),
        )
        .reset_index()
        .sort_values(["mention_count", "distinct_people", "title"], ascending=[False, False, True])
    )
    return people, titles


def iter_records(df: pd.DataFrame):
    iterator = df.iterrows()
    if tqdm is not None:
        iterator = tqdm(iterator, total=len(df), desc="Extracting people")
    return iterator


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract human people, titles, frequencies, and context snippets from CSV article text.")
    parser.add_argument("csv_path", help="Input CSV file")
    parser.add_argument("--article-col", default=None, help="Article text column name. Defaults to column B / second column.")
    parser.add_argument("--headline-col", default="headline", help="Optional headline column name for context. Default: headline")
    parser.add_argument("--outdir", default="people_title_outputs", help="Output directory")
    parser.add_argument("--method", choices=["llm", "regex"], default="llm", help="Extraction method. Default: llm")
    parser.add_argument("--model", default="gpt-4.1-mini", help="OpenAI model for --method llm")
    parser.add_argument("--max-article-chars", type=int, default=16000, help="Max article characters sent to the LLM")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for testing")
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    if args.limit:
        df = df.head(args.limit).copy()

    article_col = args.article_col or df.columns[1]
    if article_col not in df.columns:
        raise ValueError(f"Article column '{article_col}' not found. Available columns: {list(df.columns)}")
    headline_col = args.headline_col if args.headline_col in df.columns else None

    client = get_openai_client() if args.method == "llm" else None

    rows: List[Dict[str, Any]] = []
    for idx, record in iter_records(df):
        text = str(record.get(article_col, "") or "")
        headline = record.get(headline_col, "") if headline_col else ""
        if args.method == "llm":
            mentions = extract_mentions_llm(text, client=client, model=args.model, max_chars=args.max_article_chars)
        else:
            mentions = extract_mentions_regex(text)

        for mention in mentions:
            rows.append({
                "row_number": int(idx) + 2,
                "source_index": int(idx),
                "headline": headline,
                **mention,
            })

    long_df = pd.DataFrame(rows)
    long_path = outdir / "people_mentions_long.csv"
    people_path = outdir / "people_title_mentions.csv"
    titles_path = outdir / "title_frequency.csv"

    if long_df.empty:
        long_df.to_csv(long_path, index=False)
        pd.DataFrame().to_csv(people_path, index=False)
        pd.DataFrame().to_csv(titles_path, index=False)
        print("No human people found.")
        return

    people, titles = summarize(long_df)
    long_df.to_csv(long_path, index=False)
    people.to_csv(people_path, index=False)
    titles.to_csv(titles_path, index=False)

    print(f"Wrote {long_path}")
    print(f"Wrote {people_path}")
    print(f"Wrote {titles_path}")
    print("\nTop people/title mentions:")
    print(people.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
