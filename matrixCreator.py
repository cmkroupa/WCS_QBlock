import polars as pl
import re
from urllib.parse import urlparse


def extract_url_features(url):
    try:
        hostname = urlparse(url).hostname or ""
    except:
        hostname = ""

    return {
        "url_length": len(url),
        "n_dots": url.count("."),
        "n_hyphens": url.count("-"),
        "n_slash": url.count("/"),
        "is_ip": 1 if re.search(r'\d+\.\d+\.\d+\.\d+', url) else 0,
        "n_subdomains": max(0, hostname.count(".") - 1) if hostname else 0,
        "has_scam_keyword": 1 if any(
            word in url.lower() for word in ["login", "verify", "bank", "secure", "update", "pay"]) else 0
    }


def build_matrix(input_file, output_file):
    df = pl.read_csv(input_file)
    print(f"Creating matrix from {len(df)} URLs...")

    #Extract features
    feature_rows = [extract_url_features(u) for u in df["url"].to_list()]
    matrix_df = pl.DataFrame(feature_rows)

    #Attach the label
    if "label" in df.columns:
        matrix_df = matrix_df.with_columns(df["label"])

    matrix_df.write_csv(output_file)
    print(f"Success! Saved to {output_file}")


if __name__ == "__main__":
    build_matrix("raw_urls.csv", "your_matrix.csv")