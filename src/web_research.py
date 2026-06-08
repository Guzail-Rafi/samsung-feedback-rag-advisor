import os
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from ddgs import DDGS
from dotenv import load_dotenv
from langsmith import traceable

from mlflow_tracing import mlflow_span
from text_cleanup import sanitize_text
from tracing_utils import sanitize_trace_inputs, sanitize_trace_outputs


ENV_PATH = Path(__file__).resolve().parents[1] / ".env"

DEFAULT_ALLOWED_DOMAINS = {
    "apple.com",
    "amazon.ae",
    "bloomberg.com",
    "canalys.com",
    "counterpointresearch.com",
    "emaxme.com",
    "gsmarena.com",
    "gulfnews.com",
    "idc.com",
    "jumbo.ae",
    "khaleejtimes.com",
    "noon.com",
    "reuters.com",
    "samsung.com",
    "sharafdg.com",
    "statcounter.com",
    "techradar.com",
    "thenationalnews.com",
    "theverge.com",
    "tomsguide.com",
}

FOCUS_QUERIES = {
    "latest_samsung_news": "latest Samsung Galaxy product news",
    "uae_pricing_offers": "Samsung Galaxy Ultra UAE price offers trade-in",
    "competitor_iphone_offers": "Apple iPhone Pro Max offers pricing competitor",
    "current_market_trends": "current premium smartphone market trends Samsung Apple",
    "regional_market_context": "UAE GCC smartphone market Samsung Apple regional",
    "positioning_and_pricing": "Samsung Galaxy Ultra pricing positioning versus iPhone",
}


def get_allowed_domains():
    load_dotenv(dotenv_path=ENV_PATH, override=True)
    configured = os.getenv("WEB_RESEARCH_ALLOWED_DOMAINS", "").strip()

    if not configured:
        return DEFAULT_ALLOWED_DOMAINS

    return {
        domain.strip().lower()
        for domain in configured.split(",")
        if domain.strip()
    }


def domain_is_allowed(url, allowed_domains):
    hostname = (urlparse(url).hostname or "").lower().removeprefix("www.")
    return any(hostname == domain or hostname.endswith(f".{domain}") for domain in allowed_domains)


def matched_research_focus(result_text, focus, url):
    has_samsung = any(term in result_text for term in ["samsung", "galaxy"])
    has_competitor = any(term in result_text for term in ["apple", "iphone", "competitor"])
    has_commercial = any(
        term in result_text
        for term in ["price", "pricing", "offer", "deal", "discount", "trade-in", "trade in", "aed", "instalment"]
    )
    has_market = any(term in result_text for term in ["market", "trend", "share", "shipment", "sales"])
    has_region = any(
        term in result_text
        for term in ["uae", "dubai", "abu dhabi", "gcc", "gulf", "middle east", ".ae/"]
    ) or any(domain in url for domain in ["gulfnews.com", "thenationalnews.com"])
    matched = []

    if "latest_samsung_news" in focus and has_samsung:
        matched.append("latest_samsung_news")
    if "uae_pricing_offers" in focus and has_samsung and has_region and has_commercial:
        matched.append("uae_pricing_offers")
    if "competitor_iphone_offers" in focus and has_competitor and has_commercial:
        matched.append("competitor_iphone_offers")
    if "current_market_trends" in focus and (has_samsung or has_competitor) and has_market:
        matched.append("current_market_trends")
    if "regional_market_context" in focus and (has_samsung or has_competitor) and has_region and has_market:
        matched.append("regional_market_context")
    if "positioning_and_pricing" in focus and has_samsung and has_commercial:
        matched.append("positioning_and_pricing")

    return matched


def relevance_reason(matched_focus):
    focus_text = ", ".join(item.replace("_", " ") for item in matched_focus)
    return f"Source matches the controlled external research focus: {focus_text}."


def extract_date(result, snippet):
    if result.get("date"):
        return result["date"]

    match = re.match(
        r"("
        r"\d{1,2}\s+[A-Z][a-z]{2}\s+\d{4}"
        r"|[A-Z][a-z]+\s+\d{1,2},\s+\d{4}"
        r"|\d+\s+(?:day|days|week|weeks|month|months)\s+ago"
        r")",
        snippet,
    )
    return match.group(1) if match else None


def normalize_result(result, matched_focus, retrieved_at):
    url = result.get("url") or result.get("href") or ""
    title = sanitize_text(result.get("title")).strip()
    snippet = sanitize_text(result.get("body") or result.get("snippet")).strip()

    return {
        "title": title,
        "url": url,
        "snippet": snippet[:600],
        "date": extract_date(result, snippet),
        "retrieved_at": retrieved_at,
        "relevance_reason": relevance_reason(matched_focus),
        "source": result.get("source") or (urlparse(url).hostname or ""),
    }


@mlflow_span("Market Research Agent - Web Retrieval", "RETRIEVER")
@traceable(
    name="Market Research Agent - Web Retrieval",
    run_type="retriever",
    tags=["web-augmented-strategy-rag", "market-research", "web-retrieval"],
    process_inputs=sanitize_trace_inputs,
    process_outputs=sanitize_trace_outputs,
)
def search_market_evidence(query, focus, max_results=None):
    load_dotenv(dotenv_path=ENV_PATH, override=True)
    result_limit = max_results or int(os.getenv("WEB_SEARCH_MAX_RESULTS", "3"))
    max_queries = int(os.getenv("WEB_SEARCH_MAX_QUERIES", "2"))
    candidate_limit = max(result_limit * 3, 10)
    region = os.getenv("WEB_SEARCH_REGION", "ae-en")
    timelimit = os.getenv("WEB_SEARCH_TIMELIMIT", "y")
    timeout = int(os.getenv("WEB_SEARCH_TIMEOUT_SECONDS", "20"))
    allowed_domains = get_allowed_domains()
    retrieved_at = datetime.now(timezone.utc).isoformat()
    search_queries = [
        f"{query} {FOCUS_QUERIES[item]}"
        for item in focus[:max_queries]
        if item in FOCUS_QUERIES
    ] or [f"{query} Samsung smartphone market"]
    raw_results = []
    errors = []

    for search_query in search_queries:
        try:
            raw_results.extend(
                DDGS(timeout=timeout).text(
                    search_query,
                    region=region,
                    safesearch="moderate",
                    timelimit=timelimit,
                    max_results=candidate_limit,
                )
            )

            if "latest_samsung_news" in focus:
                raw_results.extend(
                    DDGS(timeout=timeout).news(
                        search_query,
                        region=region,
                        safesearch="moderate",
                        timelimit="m",
                        max_results=candidate_limit,
                    )
                )
        except Exception as error:
            errors.append(f"{error.__class__.__name__}: {error}")

    evidence = []
    seen_urls = set()

    for result in raw_results:
        url = result.get("url") or result.get("href") or ""
        result_text = f"{result.get('title', '')} {result.get('body', '')} {url}".lower()
        matched_focus = matched_research_focus(result_text, focus, url)

        if (
            not url
            or url in seen_urls
            or not domain_is_allowed(url, allowed_domains)
            or not matched_focus
        ):
            continue

        normalized = normalize_result(result, matched_focus, retrieved_at)
        if len(normalized["snippet"]) < 60:
            continue

        seen_urls.add(url)
        evidence.append(normalized)

        if len(evidence) >= result_limit:
            break

    return {
        "provider": "DDGS",
        "query_count": len(search_queries),
        "result_count": len(evidence),
        "evidence": evidence,
        "errors": errors,
        "retrieved_at": retrieved_at,
    }
