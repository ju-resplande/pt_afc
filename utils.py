import json
import os
import re

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    HarmBlockThreshold,
    HarmCategory,
)
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv, find_dotenv
import requests

load_dotenv(find_dotenv())

TEMPLATE = """
 Qual principal fato exposto no texto?
 - Extraia um trecho de até 20 palavras do texto a seguir.
 - Retorne somente a alegação e sem título

 Texto: {text}
 Alegação:
"""
QUOTES = (
    "\u0022"  # quotation mark (")
    "\u0027"  # apostrophe (')
    "\u00ab"  # left-pointing double-angle quotation mark
    "\u00bb"  # right-pointing double-angle quotation mark
    "\u2018"  # left single quotation mark
    "\u2019"  # right single quotation mark
    "\u201a"  # single low-9 quotation mark
    "\u201b"  # single high-reversed-9 quotation mark
    "\u201c"  # left double quotation mark
    "\u201d"  # right double quotation mark
    "\u201e"  # double low-9 quotation mark
    "\u201f"  # double high-reversed-9 quotation mark
    "\u2039"  # single left-pointing angle quotation mark
    "\u203a"  # single right-pointing angle quotation mark
    "\u300c"  # left corner bracket
    "\u300d"  # right corner bracket
    "\u300e"  # left white corner bracket
    "\u300f"  # right white corner bracket
    "\u301d"  # reversed double prime quotation mark
    "\u301e"  # double prime quotation mark
    "\u301f"  # low double prime quotation mark
    "\ufe41"  # presentation form for vertical left corner bracket
    "\ufe42"  # presentation form for vertical right corner bracket
    "\ufe43"  # presentation form for vertical left corner white bracket
    "\ufe44"  # presentation form for vertical right corner white bracket
    "\uff02"  # fullwidth quotation mark
    "\uff07"  # fullwidth apostrophe
    "\uff62"  # halfwidth left corner bracket
    "\uff63"  # halfwidth right corner bracket
)

emoji_pattern = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "]+",
    flags=re.UNICODE,
)

claim_extraction = PromptTemplate.from_template(TEMPLATE) | ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    # safety_settings=None,
    safety_settings={
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    },
)


def search_engine(
    query: str, locale="pt-BR", results_lang="lang_pt", n: int = 5
) -> list[dict]:
    results = requests.get(
        "https://www.googleapis.com/customsearch/v1",
        params={
            "key": os.environ["SEARCH_KEY"],
            "cx": os.environ["SEARCH_ID"],
            "gl": locale,
            # "cr": "countryBR",
            "lr": results_lang,
            "num": n,
            "q": query,
        },
    )

    try:
        results = results.json()
    except json.JSONDecodeError:
        return None

    results = results.get("items")

    if results:
        for idx, r in enumerate(results):
            clean_r = {
                "title": r.get("htmlTitle"),
                "link": r.get("link"),
                "snippet": r.get("htmlSnippet"),
            }

            pagemap = {}
            if r.get("pagemap"):
                if r.get("metags"):
                    pagemap["title"] = r["pagemap"]["metatags"][0].get("og:title")
                    pagemap["description"] = r["pagemap"]["metatags"][0].get(
                        "og:description"
                    )

                pagemap["claim_review"] = None

                if r["pagemap"].get("ClaimReview"):
                    pagemap["claim_review"] = r["pagemap"]["ClaimReview"][0].get(
                        "claimReviewed"
                    )
                    pagemap["date_review"] = r["pagemap"]["ClaimReview"][0].get(
                        "datePublished"
                    )
                else:
                    pagemap["date_review"] = None

            clean_r["pagemap"] = pagemap
            results[idx] = clean_r

    return results
