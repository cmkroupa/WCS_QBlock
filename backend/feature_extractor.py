import csv
import ipaddress
import math
import re
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Set, Tuple
from urllib.parse import urlparse

import tldextract as _tldextract_lib

SUSPICIOUS_TLDS = {
    # Original list
    "zip", "review", "country", "kim", "cricket", "science", "work", "party", "gq",
    # Heavily abused modern gTLDs
    "xyz", "top", "tk", "ml", "ga", "cf", "pw", "click", "link", "online",
    "site", "icu", "buzz", "shop", "vip", "win", "loan", "bid", "trade",
    "date", "racing", "download", "stream", "accountant", "faith", "men", "mom",
}

COMMON_TLDS = {
    "com", "org", "net", "edu", "gov", "io", "co", "info",
    "biz", "us", "uk", "ca", "de", "fr", "ru", "jp", "cn",
    "au", "it", "nl", "br", "in",
}

SENSITIVE_KEYWORDS = [
    "login", "log-in", "signin", "sign-in", "verify", "verification",
    "secure", "security", "update", "account", "auth", "password",
    "reset", "confirm",
]

TOP_BRANDS = [
    "google", "microsoft", "apple", "amazon", "facebook", "instagram",
    "whatsapp", "paypal", "netflix", "youtube", "linkedin", "x",
    "twitter", "tiktok", "snapchat", "telegram", "discord", "github",
    "dropbox", "adobe", "salesforce", "shopify", "ebay", "walmart",
    "chase", "bankofamerica", "wellsfargo", "citibank", "americanexpress",
    "capitalone", "coinbase", "binance", "airbnb", "uber", "lyft",
    "spotify", "hulu", "zoom", "slack", "okta", "cloudflare", "oracle",
    "ibm", "docusign", "intuit", "steam", "epicgames", "roblox",
    "minecraft", "playstation", "xbox", "nintendo", "samsung", "huawei",
    "xiaomi", "oppo", "vivo", "tesla", "ford", "toyota", "bmw",
    "mercedes", "nike", "adidas", "puma", "gucci", "prada", "zara",
    "ikea", "target", "costco", "bestbuy", "homedepot", "lowes",
    "fedex", "ups", "usps", "dhl", "verizon", "att", "tmobile",
    "vodafone", "airtel", "vodacom", "bloomberg", "reuters", "bbc",
    "cnn", "nytimes", "wsj", "forbes", "reddit", "pinterest", "quora",
    "notion", "figma", "canva", "atlassian", "trello", "asana",
    "mailchimp", "zendesk", "twilio", "stripe", "wise", "mastercard",
    "visa", "discover", "booking", "expedia", "tripadvisor", "doordash",
    "grubhub", "ubereats",
]

URL_SHORTENERS = {
    "bit.ly", "t.co", "tinyurl.com", "goo.gl", "ow.ly", "buff.ly",
    "short.io", "rebrand.ly", "cutt.ly", "is.gd", "v.gd", "rb.gy",
    "shorturl.at", "tiny.cc", "lnkd.in", "fb.me", "youtu.be",
    "amzn.to", "smarturl.it", "s.id", "bl.ink", "t.ly", "clck.ru",
    "qr.io", "urlz.fr",
}

SUSPICIOUS_EXTENSIONS = {
    ".exe", ".php", ".zip", ".bat", ".cmd", ".vbs", ".ps1",
    ".msi", ".dmg", ".scr", ".jar", ".hta", ".pif",
}

FEATURE_COLUMNS = [
    "tranco_rank_score",
    "in_tranco_top_10k",
    "in_tranco_top_100k",
    "domain_in_tranco_vs_subdomain",
    "is_punycode",
    "hostname_entropy",
    "levenshtein_distance_to_brand",
    "brand_in_subdomain",
    "brand_in_path",
    "hostname_digit_ratio",
    "hostname_dash_ratio",
    "number_of_subdomains",
    "tld_is_suspicious",
    "tld_length",
    "total_url_length",
    "path_length",
    "query_length",
    "number_of_path_tokens",
    "double_slash_in_path",
    "at_symbol_presence",
    "multiple_tlds_in_hostname",
    "sensitive_keyword_density",
    "hex_encoding_count",
    "port_is_non_standard",
    "dot_count",
    "dash_count",
    "equal_count",
    "ampersand_count",
    "underscore_count",
    "non_alphanumeric_ratio",
    # New features
    "hostname_is_ip",
    "is_url_shortener",
    "path_depth",
    "has_suspicious_extension",
    "vowel_consonant_ratio",
]

_TRANGO_CACHE: Optional[Tuple[Dict[str, int], Set[str], Set[str]]] = None

# Module-level extractor. Uses the cached PSL on disk (filelock-safe for multiprocessing).
# Each worker subprocess imports this module and gets its own instance backed by the same
# shared cache file — tldextract handles concurrent access with a filelock.
_TLD_EXTRACTOR = _tldextract_lib.TLDExtract()


def _load_tranco_index() -> Tuple[Dict[str, int], Set[str], Set[str]]:
    global _TRANGO_CACHE
    if _TRANGO_CACHE is not None:
        return _TRANGO_CACHE

    tranco_path = Path(__file__).resolve().parent / "artifacts" / "tranco_full.csv"
    rank_by_domain: Dict[str, int] = {}
    top_10k: Set[str] = set()
    top_100k: Set[str] = set()

    if tranco_path.exists():
        with open(tranco_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 2:
                    continue
                try:
                    rank = int(row[0])
                except ValueError:
                    continue
                domain = row[1].strip().lower()
                if not domain:
                    continue
                rank_by_domain[domain] = rank
                if rank < 10_000:
                    top_10k.add(domain)
                if rank < 100_000:
                    top_100k.add(domain)

    _TRANGO_CACHE = (rank_by_domain, top_10k, top_100k)
    return _TRANGO_CACHE


def _safe_urlparse(url: str):
    normalized = url.strip()
    if not normalized.startswith(("http://", "https://")):
        normalized = f"http://{normalized}"
    return urlparse(normalized)


def _hostname_entropy(hostname: str) -> float:
    if not hostname:
        return 0.0
    counts: Dict[str, int] = {}
    for ch in hostname:
        counts[ch] = counts.get(ch, 0) + 1
    entropy = 0.0
    total = len(hostname)
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr.append(min(curr[-1] + 1, prev[j] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def _embedded_tranco_domain(hostname: str, root_domain: str, tranco_domains: Mapping[str, int]) -> float:
    labels = [label for label in hostname.lower().split(".") if label]
    n = len(labels)
    for i in range(0, n - 1):
        for j in range(i + 2, n + 1):
            candidate = ".".join(labels[i:j])
            if candidate == root_domain:
                continue
            if candidate in tranco_domains:
                return 1.0
    return 0.0


def _min_brand_levenshtein(hostname: str, brands: List[str], exclude_labels: Set[str] = frozenset()) -> float:
    """
    Returns the minimum Levenshtein distance between any meaningful hostname label
    and any brand name, ignoring exact matches (d=0 means it IS the brand — legitimate).

    Checks every label except those in exclude_labels (the TLD suffix parts, e.g. {"com"},
    {"co", "uk"}) to prevent short TLD labels like "com" from producing artificially low
    distances to short brands like "x" or "ibm" and making the feature constant.

    Catches paypal.security-update.com style attacks by checking all non-TLD labels.
    """
    if not hostname:
        return float(max(len(b) for b in brands))

    labels = [re.sub(r"[^a-z0-9]", "", label) for label in hostname.lower().split(".")]
    non_empty = [label for label in labels if label and label not in exclude_labels]
    if not non_empty:
        return float(max(len(b) for b in brands))

    distances = []
    for label in non_empty:
        for brand in brands:
            d = _levenshtein(label, brand)
            if d > 0:
                distances.append(d)
    if not distances:
        return 0.0
    return float(min(distances))


def _is_ip_address(hostname: str) -> bool:
    try:
        ipaddress.ip_address(hostname)
        return True
    except ValueError:
        return False


def _vowel_consonant_ratio(hostname: str) -> float:
    letters = [c for c in hostname.lower() if c.isalpha()]
    if not letters:
        return 0.0
    vowels = sum(1 for c in letters if c in "aeiou")
    consonants = len(letters) - vowels
    if consonants == 0:
        return float(vowels)
    return vowels / consonants


def extract_url_features(url: str) -> Dict[str, float]:
    parsed = _safe_urlparse(url)
    hostname = (parsed.hostname or "").lower()
    path = parsed.path or ""
    query = parsed.query or ""
    full_url = url.strip()

    total_length = len(full_url)
    path_length = len(path)
    query_length = len(query)
    hostname_length = len(hostname) if hostname else 0

    # --- IP detection (must happen before TLD extraction) ---
    hostname_is_ip = 1.0 if (hostname and _is_ip_address(hostname)) else 0.0

    # --- TLD / subdomain extraction via Public Suffix List ---
    suffix_labels: Set[str] = set()
    if hostname_is_ip or not hostname:
        root_domain = hostname
        tld = ""
        subdomain = ""
    else:
        ext = _TLD_EXTRACTOR(hostname)
        root_domain = f"{ext.domain}.{ext.suffix}" if ext.suffix else ext.domain
        tld = ext.suffix.split(".")[-1] if ext.suffix else ""
        subdomain = ext.subdomain
        # Labels that are part of the public suffix (e.g. {"com"} or {"co","uk"})
        # — excluded from brand Levenshtein checks to prevent false positives.
        if ext.suffix:
            suffix_labels = set(ext.suffix.lower().split("."))

    rank_by_domain, tranco_top_10k, tranco_top_100k = _load_tranco_index()
    tranco_rank = rank_by_domain.get(root_domain)
    tranco_max = 5_600_000
    if tranco_rank is None or tranco_rank <= 0:
        tranco_rank_score = -1.0
    else:
        tranco_rank_score = 1.0 - 2.0 * (math.log(tranco_rank) / math.log(tranco_max))

    in_tranco_top_10k = 1.0 if root_domain in tranco_top_10k else 0.0
    in_tranco_top_100k = 1.0 if root_domain in tranco_top_100k else 0.0
    domain_in_tranco_vs_subdomain = 1.0 if (root_domain not in rank_by_domain and _embedded_tranco_domain(hostname, root_domain, rank_by_domain) > 0.0) else 0.0

    is_punycode = 1.0 if any(label.startswith("xn--") for label in hostname.split(".") if label) else 0.0
    hostname_entropy = _hostname_entropy(hostname)
    levenshtein_distance_to_brand = _min_brand_levenshtein(hostname, TOP_BRANDS, suffix_labels)

    brand_in_subdomain = 1.0 if any(brand in subdomain for brand in TOP_BRANDS) else 0.0
    brand_in_path = 1.0 if any(brand in path.lower() for brand in TOP_BRANDS) else 0.0
    hostname_digit_ratio = (sum(ch.isdigit() for ch in hostname) / hostname_length) if hostname_length else 0.0
    hostname_dash_ratio = (hostname.count("-") / hostname_length) if hostname_length else 0.0
    number_of_subdomains = float(len(subdomain.split(".")) if subdomain else 0)
    tld_is_suspicious = 1.0 if tld in SUSPICIOUS_TLDS else 0.0
    tld_length = float(len(tld))

    path_tokens = [token for token in re.split(r"[\/_.-]", path) if token]
    number_of_path_tokens = float(len(path_tokens))
    double_slash_in_path = 1.0 if "//" in path else 0.0
    at_symbol_presence = 1.0 if "@" in full_url else 0.0
    multiple_tlds_in_hostname = 1.0 if any(label in COMMON_TLDS for label in hostname.split(".")[:-1]) else 0.0

    lower_url = full_url.lower()
    # Count distinct keywords present (0–len(SENSITIVE_KEYWORDS)), normalised to 0–1.
    # Dividing by URL length (previous formula) made short domains with one keyword
    # score higher than long phishing URLs with many keywords — now URL-length-independent.
    distinct_keywords_found = sum(1 for kw in SENSITIVE_KEYWORDS if kw in lower_url)
    sensitive_keyword_density = distinct_keywords_found / len(SENSITIVE_KEYWORDS)
    # Normalise by URL length so a 1000-char legitimate URL with 50 encoded chars
    # doesn't dwarf a 60-char phishing URL with 10 encoded chars.
    raw_hex_count = len(re.findall(r"%[0-9a-fA-F]{2}", full_url))
    hex_encoding_count = float(raw_hex_count / total_length) if total_length else 0.0
    try:
        parsed_port = parsed.port
    except ValueError:
        parsed_port = -1
    port_is_non_standard = 1.0 if parsed_port not in (None, 80, 443) else 0.0

    dot_count = float(full_url.count("."))
    dash_count = float(full_url.count("-"))
    equal_count = float(full_url.count("="))
    ampersand_count = float(full_url.count("&"))
    underscore_count = float(full_url.count("_"))
    special_chars = sum(not ch.isalnum() for ch in full_url)
    non_alphanumeric_ratio = (special_chars / total_length) if total_length else 0.0

    # --- New features ---
    is_url_shortener = 1.0 if root_domain.lower() in URL_SHORTENERS else 0.0
    path_depth = float(len([seg for seg in path.split("/") if seg]))
    lower_path = path.lower().rstrip("/")
    has_suspicious_extension = 1.0 if any(lower_path.endswith(ext) for ext in SUSPICIOUS_EXTENSIONS) else 0.0
    vowel_consonant_ratio = _vowel_consonant_ratio(hostname)

    return {
        "tranco_rank_score": float(tranco_rank_score),
        "in_tranco_top_10k": float(in_tranco_top_10k),
        "in_tranco_top_100k": float(in_tranco_top_100k),
        "domain_in_tranco_vs_subdomain": float(domain_in_tranco_vs_subdomain),
        "is_punycode": float(is_punycode),
        "hostname_entropy": float(hostname_entropy),
        "levenshtein_distance_to_brand": float(levenshtein_distance_to_brand),
        "brand_in_subdomain": float(brand_in_subdomain),
        "brand_in_path": float(brand_in_path),
        "hostname_digit_ratio": float(hostname_digit_ratio),
        "hostname_dash_ratio": float(hostname_dash_ratio),
        "number_of_subdomains": float(number_of_subdomains),
        "tld_is_suspicious": float(tld_is_suspicious),
        "tld_length": float(tld_length),
        "total_url_length": float(total_length),
        "path_length": float(path_length),
        "query_length": float(query_length),
        "number_of_path_tokens": float(number_of_path_tokens),
        "double_slash_in_path": float(double_slash_in_path),
        "at_symbol_presence": float(at_symbol_presence),
        "multiple_tlds_in_hostname": float(multiple_tlds_in_hostname),
        "sensitive_keyword_density": float(sensitive_keyword_density),
        "hex_encoding_count": float(hex_encoding_count),
        "port_is_non_standard": float(port_is_non_standard),
        "dot_count": float(dot_count),
        "dash_count": float(dash_count),
        "equal_count": float(equal_count),
        "ampersand_count": float(ampersand_count),
        "underscore_count": float(underscore_count),
        "non_alphanumeric_ratio": float(non_alphanumeric_ratio),
        "hostname_is_ip": float(hostname_is_ip),
        "is_url_shortener": float(is_url_shortener),
        "path_depth": float(path_depth),
        "has_suspicious_extension": float(has_suspicious_extension),
        "vowel_consonant_ratio": float(vowel_consonant_ratio),
    }
