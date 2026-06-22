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
from web_strategy_policy import classify_product_lifecycle


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
    "google.com",
    "idc.com",
    "jumbo.ae",
    "khaleejtimes.com",
    "mediaoffice.ae",
    "mi.com",
    "noon.com",
    "reuters.com",
    "samsung.com",
    "sharafdg.com",
    "statcounter.com",
    "techradar.com",
    "thenationalnews.com",
    "theverge.com",
    "timeoutdubai.com",
    "tomsguide.com",
    "uae.gov.ae",
    "visitdubai.com",
}

FOCUS_QUERIES = {
    "product_lifecycle_verification": "Samsung Galaxy model official availability launch support",
    "competitor_apple_playbook": "site:apple.com/newsroom iPhone 17 Pro launch camera storage Apple Intelligence Trade In financing",
    "competitor_apple_sales_result": "site:counterpointresearch.com Apple premium smartphone sales growth 2025 iPhone",
    "competitor_pixel_playbook": "site:blog.google/products/pixel/made-by-google-2025 Pixel AI camera Gemini",
    "competitor_pixel_sales_result": "site:counterpointresearch.com Google Pixel record sales September 2025 premium",
    "latest_samsung_news": "latest Samsung Galaxy product news",
    "uae_pricing_offers": "Samsung Galaxy Ultra UAE current price offers trade-in instalments",
    "uae_retail_events": "UAE Dubai retail events promotion calendar Ramadan Eid Dubai Shopping Festival White Friday",
    "competitor_iphone_offers": "Apple iPhone Pro Max offers pricing competitor",
    "current_market_trends": "current premium smartphone market trends Samsung Apple",
    "regional_market_context": "UAE GCC smartphone market Samsung Apple regional",
    "positioning_and_pricing": "Samsung Galaxy Ultra pricing positioning versus iPhone",
}


def build_focus_query(focus_item, lifecycle, year):
    current_model = lifecycle["current_model"]
    requested_model = lifecycle.get("requested_model")

    if focus_item == "product_lifecycle_verification" and requested_model:
        return f"Samsung {requested_model} official Samsung UAE launch availability support"
    if focus_item in {
        "competitor_apple_playbook",
        "competitor_apple_sales_result",
        "competitor_pixel_playbook",
        "competitor_pixel_sales_result",
    }:
        return FOCUS_QUERIES[focus_item]
    if focus_item == "uae_pricing_offers":
        return f"Samsung {current_model} UAE current price offers trade-in instalments {year}"
    if focus_item == "positioning_and_pricing":
        return f"Samsung {current_model} UAE pricing positioning versus iPhone {year}"
    if focus_item == "latest_samsung_news":
        return f"latest Samsung {current_model} product news {year}"

    return f"{FOCUS_QUERIES[focus_item]} {year}"


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


def model_tokens(lifecycle):
    generation = lifecycle.get("requested_generation")
    if generation is None:
        return []
    return [
        f"s{generation} ultra",
        f"s{generation}-ultra",
        f"galaxy s{generation}",
    ]


def matched_research_focus(result_text, focus, url, lifecycle):
    has_samsung = any(term in result_text for term in ["samsung", "galaxy"])
    has_competitor = any(term in result_text for term in ["apple", "iphone", "competitor"])
    has_apple = any(term in result_text for term in ["apple", "iphone"])
    has_pixel = any(term in result_text for term in ["google pixel", "pixel 9", "pixel 10"])
    has_marketing_tactic = any(
        term in result_text
        for term in [
            "launch",
            "camera",
            "ai",
            "trade-in",
            "trade in",
            "financing",
            "storage",
            "ecosystem",
            "bundle",
            "creator",
        ]
    )
    has_sales_result = any(
        term in result_text
        for term in [
            "sales",
            "shipments",
            "growth",
            "market share",
            "record",
            "revenue",
            "sell-in",
            "sell-out",
        ]
    )
    is_official_apple = "apple.com" in url
    is_official_google = "google.com" in url
    is_outcome_source = any(
        domain in url
        for domain in [
            "counterpointresearch.com",
            "idc.com",
            "canalys.com",
            "apple.com",
            "google.com",
            "mi.com",
        ]
    )
    has_commercial = any(
        term in result_text
        for term in ["price", "pricing", "offer", "deal", "discount", "trade-in", "trade in", "aed", "instalment"]
    )
    has_market = any(term in result_text for term in ["market", "trend", "share", "shipment", "sales"])
    has_region = any(
        term in result_text
        for term in [
            "uae",
            "dubai",
            "abu dhabi",
            "gcc",
            "gulf",
            "middle east",
            ".ae/",
            "/ae/",
        ]
    ) or any(domain in url for domain in ["gulfnews.com", "thenationalnews.com"])
    has_event = any(
        term in result_text
        for term in [
            "shopping festival",
            "retail event",
            "ramadan",
            "eid",
            "white friday",
            "black friday",
            "promotion",
            "sale",
        ]
    )
    matched = []
    has_requested_model = any(token in result_text for token in model_tokens(lifecycle))

    if "product_lifecycle_verification" in focus and has_samsung and has_requested_model:
        matched.append("product_lifecycle_verification")
    if (
        "competitor_apple_playbook" in focus
        and has_apple
        and has_marketing_tactic
        and is_official_apple
    ):
        matched.append("competitor_apple_playbook")
    if (
        "competitor_pixel_playbook" in focus
        and has_pixel
        and has_marketing_tactic
        and is_official_google
    ):
        matched.append("competitor_pixel_playbook")
    if (
        "competitor_apple_sales_result" in focus
        and has_apple
        and has_sales_result
        and is_outcome_source
    ):
        matched.append("competitor_apple_sales_result")
    if (
        "competitor_pixel_sales_result" in focus
        and has_pixel
        and has_sales_result
        and is_outcome_source
    ):
        matched.append("competitor_pixel_sales_result")
    if "latest_samsung_news" in focus and has_samsung:
        matched.append("latest_samsung_news")
    if "uae_pricing_offers" in focus and has_samsung and has_region and has_commercial:
        matched.append("uae_pricing_offers")
    if "uae_retail_events" in focus and has_region and has_event:
        matched.append("uae_retail_events")
    if "competitor_iphone_offers" in focus and has_competitor and has_commercial:
        matched.append("competitor_iphone_offers")
    if "current_market_trends" in focus and (has_samsung or has_competitor) and has_market:
        matched.append("current_market_trends")
    if "regional_market_context" in focus and (has_samsung or has_competitor) and has_region and has_market:
        matched.append("regional_market_context")
    if "positioning_and_pricing" in focus and has_samsung and has_commercial:
        matched.append("positioning_and_pricing")

    return matched


def build_product_verification(lifecycle, evidence):
    if lifecycle.get("requested_generation") is None:
        return {
            "status": "not_applicable",
            "official_source_found": False,
            "explanation": "No explicit Galaxy S generation was supplied.",
        }

    tokens = model_tokens(lifecycle)
    official_matches = [
        item
        for item in evidence
        if "samsung.com" in item.get("url", "")
        and any(
            token in f"{item.get('title', '')} {item.get('url', '')} {item.get('snippet', '')}".lower()
            for token in tokens
        )
    ]

    if official_matches:
        status = (
            "verified_current_product"
            if lifecycle["lifecycle"] == "current_product"
            else "verified_existing_previous_product"
            if lifecycle["lifecycle"] == "previous_product"
            else "official_reference_found_review_lifecycle"
        )
        explanation = (
            f"An official Samsung source was found for {lifecycle['requested_model']}. "
            f"The model is handled as {lifecycle['lifecycle'].replace('_', ' ')}."
        )
    elif lifecycle["lifecycle"] == "future_product":
        status = "unverified_future_product"
        explanation = (
            f"No official Samsung source was retrieved confirming {lifecycle['requested_model']}; "
            "it remains a hypothetical future planning target."
        )
    else:
        status = "existing_product_not_officially_verified_in_this_search"
        explanation = (
            f"The search did not retrieve an official Samsung source for "
            f"{lifecycle['requested_model']}; recommendations should remain provisional."
        )

    return {
        "status": status,
        "official_source_found": bool(official_matches),
        "official_sources": [item["url"] for item in official_matches],
        "explanation": explanation,
    }


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
    configured_max_queries = int(os.getenv("WEB_SEARCH_MAX_QUERIES", "2"))
    candidate_limit = max(result_limit * 3, 10)
    region = os.getenv("WEB_SEARCH_REGION", "ae-en")
    timelimit = os.getenv("WEB_SEARCH_TIMELIMIT", "y")
    timeout = int(os.getenv("WEB_SEARCH_TIMEOUT_SECONDS", "20"))
    allowed_domains = get_allowed_domains()
    retrieved_at = datetime.now(timezone.utc).isoformat()
    lifecycle = classify_product_lifecycle(query)
    current_year = datetime.now(timezone.utc).year
    max_queries = (
        max(configured_max_queries, min(8, len(focus)))
        if "product_lifecycle_verification" in focus and len(focus) > 1
        else configured_max_queries
    )
    search_specs = [
        (item, build_focus_query(item, lifecycle, current_year))
        for item in focus[:max_queries]
        if item in FOCUS_QUERIES
    ] or [
        (
            "general_market_research",
            f"{query} Samsung smartphone market {current_year}",
        )
    ]
    search_queries = [search_query for _, search_query in search_specs]
    raw_results = []
    errors = []

    for focus_item, search_query in search_specs:
        try:
            search_timelimit = (
                None
                if focus_item == "product_lifecycle_verification"
                else timelimit
            )
            text_results = DDGS(timeout=timeout).text(
                search_query,
                region=region,
                safesearch="moderate",
                timelimit=search_timelimit,
                max_results=candidate_limit,
            )
            raw_results.extend(
                {**result, "_search_focus": focus_item}
                for result in text_results
            )

            if "latest_samsung_news" in focus:
                news_results = DDGS(timeout=timeout).news(
                    search_query,
                    region=region,
                    safesearch="moderate",
                    timelimit="m",
                    max_results=candidate_limit,
                )
                raw_results.extend(
                    {**result, "_search_focus": focus_item}
                    for result in news_results
                )
        except Exception as error:
            errors.append(f"{error.__class__.__name__}: {error}")

    evidence = []
    seen_urls = set()
    per_focus_counts = {}
    max_per_focus = max(1, (result_limit + len(search_specs) - 1) // len(search_specs))

    for result in raw_results:
        url = result.get("url") or result.get("href") or ""
        result_text = f"{result.get('title', '')} {result.get('body', '')} {url}".lower()
        matched_focus = matched_research_focus(result_text, focus, url, lifecycle)
        search_focus = result.get("_search_focus", "general_market_research")

        if (
            not url
            or url in seen_urls
            or not domain_is_allowed(url, allowed_domains)
            or not matched_focus
            or (
                search_focus != "general_market_research"
                and search_focus not in matched_focus
            )
            or per_focus_counts.get(search_focus, 0) >= max_per_focus
        ):
            continue

        normalized = normalize_result(result, matched_focus, retrieved_at)
        if len(normalized["snippet"]) < 60:
            continue

        seen_urls.add(url)
        per_focus_counts[search_focus] = per_focus_counts.get(search_focus, 0) + 1
        evidence.append(normalized)

        if len(evidence) >= result_limit:
            break

    product_verification = build_product_verification(lifecycle, evidence)
    lifecycle = {**lifecycle, "verification": product_verification}

    return {
        "provider": "DDGS",
        "product_lifecycle": lifecycle,
        "product_verification": product_verification,
        "queries": search_queries,
        "query_count": len(search_queries),
        "result_count": len(evidence),
        "evidence": evidence,
        "errors": errors,
        "retrieved_at": retrieved_at,
    }
