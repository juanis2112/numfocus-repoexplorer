#!/usr/bin/env python3
"""
NumFocus Project Data Collector
================================
Fetches GitHub metrics for the 63 NumFocus sponsored projects and
saves the result as a parquet file at Data/parquet/numfocus_projects.parquet.

Usage:
    export GITHUB_TOKEN=your_personal_access_token
    python collect_data.py

The script uses the GitHub REST API (v3). A fine-grained or classic personal
access token with *public_repo* scope is sufficient (no write access needed).
Without a token the API is limited to 60 req/hour; with a token it's 5,000.
"""

import json
import os
import time
import logging
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEED_FILE   = Path("Data/projects_seed.json")
OUT_PARQUET          = Path("Data/parquet/numfocus_projects.parquet")
OUT_SECURITY_PARQUET = Path("Data/parquet/numfocus_security.parquet")
OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)

SCORECARD_API = "https://api.securityscorecards.dev"

# All checks returned by the Scorecard API (v4), in display order.
SCORECARD_CHECKS = [
    "Binary-Artifacts",
    "Branch-Protection",
    "CI-Best-Practices",
    "Code-Review",
    "Contributors",
    "Dangerous-Workflow",
    "Dependency-Update-Tool",
    "Fuzzing",
    "License",
    "Maintained",
    "Packaging",
    "Pinned-Dependencies",
    "SAST",
    "Security-Policy",
    "Signed-Releases",
    "Token-Permissions",
    "Vulnerabilities",
]

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "").strip()
HEADERS = {
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}
if GITHUB_TOKEN:
    HEADERS["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    log.info("GitHub token found — using authenticated requests (5,000 req/hr limit).")
else:
    log.warning("No GITHUB_TOKEN env var — using unauthenticated requests (60 req/hr limit).")

BASE_URL = "https://api.github.com"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get(url: str, params: Optional[dict] = None, retries: int = 3) -> Optional[dict]:
    """GET a GitHub API endpoint with retry / rate-limit handling."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
        except requests.exceptions.RequestException as exc:
            log.warning("Request error (%s): %s", url, exc)
            time.sleep(2 ** attempt)
            continue

        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 403:
            reset_ts = int(resp.headers.get("X-RateLimit-Reset", 0))
            wait = max(reset_ts - int(time.time()), 0) + 5
            log.warning("Rate-limited. Waiting %ds …", wait)
            time.sleep(wait)
            continue
        if resp.status_code == 404:
            log.warning("404 Not Found: %s", url)
            return None
        log.warning("HTTP %s for %s", resp.status_code, url)
        time.sleep(2 ** attempt)
    return None


def fetch_repo(owner_repo: str) -> dict:
    """Return the /repos/{owner}/{repo} payload, or {} on failure."""
    data = _get(f"{BASE_URL}/repos/{owner_repo}")
    return data if isinstance(data, dict) else {}


def fetch_contributor_count(owner_repo: str) -> Optional[int]:
    """
    Return total contributor count via the contributors list endpoint
    (paginated; last-page link gives total pages × 30 = approx count).
    """
    url = f"{BASE_URL}/repos/{owner_repo}/contributors"
    params = {"per_page": 1, "anon": "false"}
    resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
    if resp.status_code != 200:
        return None
    # Parse Link header to find total count
    link = resp.headers.get("Link", "")
    if 'rel="last"' in link:
        import re
        m = re.search(r'page=(\d+)>; rel="last"', link)
        if m:
            return int(m.group(1))
    # If no "last" link, there's only 1 page with per_page=1 → 1 contributor
    items = resp.json()
    return len(items) if isinstance(items, list) else None


def has_file(owner_repo: str, path: str) -> bool:
    """True if a file exists at the given path in the default branch."""
    url = f"{BASE_URL}/repos/{owner_repo}/contents/{path}"
    resp = requests.get(url, headers=HEADERS, timeout=20)
    return resp.status_code == 200


# Common license filename variants found in the wild.
# GitHub's Contents API is case-sensitive, so we probe both cases.
# Also covers projects that put licenses in a licenses/ subdirectory (e.g. TARDIS).
_LICENSE_PATHS = [
    # Root — uppercase (most common)
    "LICENSE", "LICENSE.txt", "LICENSE.md", "LICENSE.rst",
    # Root — lowercase (Cantera, GDAL, etc.)
    "license", "license.txt", "license.md", "license.rst",
    # British spelling
    "LICENCE", "LICENCE.txt", "LICENCE.md",
    "licence", "licence.txt",
    # COPYING variants (GPL-style)
    "COPYING", "COPYING.txt", "COPYING.md", "COPYING.rst",
    # Compound names
    "LICENSE-MIT", "LICENSE-APACHE", "LICENSE.Apache-2.0",
    # licenses/ subdirectory (TARDIS uses licenses/LICENSE.rst)
    "licenses/LICENSE", "licenses/LICENSE.txt", "licenses/LICENSE.md",
    "licenses/LICENSE.rst", "licenses/license.rst", "licenses/COPYING",
]

def fetch_license_fallback(owner_repo: str) -> Optional[str]:
    """
    When the GitHub repo API returns no license, probe the repo root for
    common license filenames via the Contents API.

    Returns the SPDX ID if GitHub can identify it, "OTHER" if a license
    file is found but unclassified, or None if nothing is found.
    """
    for path in _LICENSE_PATHS:
        url = f"{BASE_URL}/repos/{owner_repo}/contents/{path}"
        resp = requests.get(url, headers=HEADERS, timeout=20)
        if resp.status_code == 200:
            # The Contents API sometimes includes a license sub-object
            try:
                data = resp.json()
                if isinstance(data, dict):
                    # Try to get SPDX from the file metadata (rare but possible)
                    spdx = (data.get("license") or {}).get("spdx_id")
                    if spdx and spdx not in ("NOASSERTION", ""):
                        log.info("  License fallback: found %s via %s", spdx, path)
                        return spdx
            except Exception:
                pass
            log.info("  License fallback: found license file '%s' (marking OTHER)", path)
            return "OTHER"
    return None


def check_community_files(owner_repo: str) -> dict:
    """
    Use the GitHub community profile endpoint to check for:
    README, CONTRIBUTING, CODE_OF_CONDUCT, license, issue templates, PR template.

    NOTE: The community profile API is unreliable for SECURITY.md — it misses
    many valid files. We override `has_security_policy` with a direct file
    check across all three standard locations GitHub recognises.
    """
    url = f"{BASE_URL}/repos/{owner_repo}/community/profile"
    data = _get(url)
    if not isinstance(data, dict):
        return {}

    files = data.get("files", {}) or {}
    result = {
        "has_readme":                files.get("readme") is not None,
        "has_contributing":          files.get("contributing") is not None,
        "has_code_of_conduct":       files.get("code_of_conduct") is not None,
        "has_license":               files.get("license") is not None,
        "has_security_policy":       False,   # overridden below
        "has_issue_template":        files.get("issue_template") is not None,
        "has_pull_request_template": files.get("pull_request_template") is not None,
        "health_percentage":         data.get("health_percentage"),
        "description":               data.get("description"),
    }

    # GitHub's community API misses many SECURITY.md files.
    # Directly probe the three locations GitHub itself searches.
    security_paths = [
        "SECURITY.md",
        ".github/SECURITY.md",
        "docs/SECURITY.md",
    ]
    for path in security_paths:
        if has_file(owner_repo, path):
            result["has_security_policy"] = True
            break

    return result


def fetch_release_downloads(owner_repo: str, max_releases: int = 5) -> int:
    """Sum download counts across the most recent `max_releases` releases."""
    url = f"{BASE_URL}/repos/{owner_repo}/releases"
    data = _get(url, params={"per_page": max_releases})
    if not isinstance(data, list):
        return 0
    total = 0
    for release in data:
        for asset in release.get("assets", []):
            total += asset.get("download_count", 0)
    return total


def fetch_languages(owner_repo: str) -> dict:
    """Return {language: bytes} dict from the /languages endpoint."""
    url = f"{BASE_URL}/repos/{owner_repo}/languages"
    data = _get(url)
    return data if isinstance(data, dict) else {}


def fetch_org(org: str) -> dict:
    """Return the /orgs/{org} payload, or {} on failure."""
    data = _get(f"{BASE_URL}/orgs/{org}")
    return data if isinstance(data, dict) else {}


def fetch_org_top_repos(org: str, n: int = 10) -> list:
    """Return the top-n public repos in an org, sorted by stars."""
    data = _get(
        f"{BASE_URL}/orgs/{org}/repos",
        params={"sort": "stars", "direction": "desc", "per_page": n, "type": "public"},
    )
    return data if isinstance(data, list) else []


def fetch_scorecard(owner_repo: str) -> Optional[dict]:
    """
    Fetch OpenSSF Scorecard results from the public Scorecard REST API.
    Returns a flat dict {check_name: score, ..., "Total_Score": float}
    or None if the project is not in the Scorecard database.

    API docs: https://api.securityscorecards.dev
    """
    url = f"{SCORECARD_API}/projects/github.com/{owner_repo}"
    try:
        resp = requests.get(url, timeout=30)
    except requests.exceptions.RequestException as exc:
        log.warning("Scorecard request error for %s: %s", owner_repo, exc)
        return None

    if resp.status_code == 404:
        log.info("  Scorecard: not indexed for %s", owner_repo)
        return None
    if resp.status_code != 200:
        log.warning("  Scorecard HTTP %s for %s", resp.status_code, owner_repo)
        return None

    data = resp.json()
    result: dict = {
        "github_repo":   owner_repo,
        "scorecard_date": data.get("date"),
        "Total_Score":   data.get("score"),   # 0–10 float
    }
    # Flatten individual check scores
    for check in data.get("checks", []):
        name  = check.get("name", "").replace("-", "_")   # e.g. "Binary_Artifacts"
        score = check.get("score", -1)                    # -1 = not applicable
        result[name] = score

    return result


def bus_factor_estimate(owner_repo: str, top_n: int = 2, threshold: float = 0.5) -> Optional[int]:
    """
    Very rough bus-factor proxy: number of contributors who collectively
    account for >= 50% of commits (limited to top 100 contributors).
    Returns None if data unavailable.
    """
    url = f"{BASE_URL}/repos/{owner_repo}/contributors"
    data = _get(url, params={"per_page": 100})
    if not isinstance(data, list) or not data:
        return None
    totals = [c.get("contributions", 0) for c in data]
    grand_total = sum(totals)
    if grand_total == 0:
        return None
    cumulative = 0
    for i, c in enumerate(totals, 1):
        cumulative += c
        if cumulative / grand_total >= threshold:
            return i
    return len(totals)


# ---------------------------------------------------------------------------
# Main collection loop
# ---------------------------------------------------------------------------

def collect_org_project(proj: dict) -> dict:
    """
    Collect aggregate GitHub metrics for projects that span a whole GitHub
    organisation (e.g. rOpenSci, Bioconductor) rather than a single repo.
    Stars, forks, and open issues are summed across the top-10 repos by stars.
    """
    org = proj.get("github_org", "")
    log.info("Collecting org  %s  (%s)", proj["name"], org)

    row: dict = {
        "name":           proj["name"],
        "numfocus_slug":  proj.get("numfocus_slug", ""),
        "numfocus_url":   proj.get("numfocus_url", ""),
        "website":        proj.get("website", ""),
        "github_repo":    "",
        "github_org":     org,
        "is_org":         True,
        "language_tags":  "|".join(proj.get("language_tags", [])),
        "feature_tags":   "|".join(proj.get("feature_tags", [])),
        "industry_tags":  "|".join(proj.get("industry_tags", [])),
        "sponsored_since": proj.get("sponsored_since"),
        "html_url":       f"https://github.com/{org}",
        "description":    None,
        "primary_language": None,
        "license":        None,
        "stargazers_count": None,
        "forks_count":    None,
        "open_issues_count": None,
        "watchers_count": None,
        "subscribers_count": None,
        "size_kb":        None,
        "created_at":     None,
        "updated_at":     None,
        "pushed_at":      None,
        "is_fork":        False,
        "default_branch": None,
        "topics":         None,
        "has_wiki":       None,
        "has_discussions": None,
        "contributor_count": None,
        "bus_factor":     None,
        "release_downloads": None,
        "top_languages":  None,
        "public_repos":   None,
        # health fields not meaningful at org level
        "has_readme":     None,
        "has_contributing": None,
        "has_code_of_conduct": None,
        "has_license":    None,
        "has_security_policy": None,
        "has_issue_template": None,
        "has_pull_request_template": None,
        "health_percentage": None,
    }

    if not org:
        return row

    org_data = fetch_org(org)
    if org_data:
        row.update({
            "description":  org_data.get("description"),
            "html_url":     org_data.get("html_url", f"https://github.com/{org}"),
            "created_at":   org_data.get("created_at"),
            "public_repos": org_data.get("public_repos"),
        })

    # Aggregate stats across top repos
    top_repos = fetch_org_top_repos(org)
    if top_repos:
        row["stargazers_count"]  = sum(r.get("stargazers_count", 0) for r in top_repos)
        row["forks_count"]       = sum(r.get("forks_count", 0)       for r in top_repos)
        row["open_issues_count"] = sum(r.get("open_issues_count", 0) for r in top_repos)
        langs = [r.get("language") for r in top_repos if r.get("language")]
        if langs:
            row["primary_language"] = max(set(langs), key=langs.count)
            row["top_languages"]    = "|".join(dict.fromkeys(l for l in langs if l))
        most_recent = max(
            (r.get("pushed_at") for r in top_repos if r.get("pushed_at")),
            default=None,
        )
        row["pushed_at"] = most_recent

    log.info("  Org %s: %d repos, %s stars (top-10 aggregate)",
             org, row["public_repos"] or 0, row["stargazers_count"] or 0)
    return row


def collect_project(proj: dict) -> dict:
    """
    Given a project seed dict, fetch all GitHub metrics and return
    a combined row dict ready for DataFrame construction.
    Dispatches to collect_org_project() for org-level entries.
    """
    # --- Dispatch: org-level projects ---
    if proj.get("github_org"):
        return collect_org_project(proj)

    repo_path = proj.get("github_repo", "")
    log.info("Collecting  %s  (%s)", proj["name"], repo_path)

    row: dict = {
        # from seed
        "name":              proj["name"],
        "numfocus_slug":     proj.get("numfocus_slug", ""),
        "numfocus_url":      proj.get("numfocus_url", ""),
        "website":           proj.get("website", ""),
        "github_repo":       repo_path,
        "github_org":        "",        # only set for org-level entries
        "is_org":            False,
        "language_tags":     "|".join(proj.get("language_tags", [])),
        "feature_tags":      "|".join(proj.get("feature_tags", [])),
        "industry_tags":     "|".join(proj.get("industry_tags", [])),
        "sponsored_since":   proj.get("sponsored_since"),
        "public_repos":      None,      # only meaningful for orgs
        # GitHub fields (default null)
        "html_url":          None,
        "description":       None,
        "primary_language":  None,
        "license":           None,
        "stargazers_count":  None,
        "forks_count":       None,
        "open_issues_count": None,
        "watchers_count":    None,
        "subscribers_count": None,
        "size_kb":           None,
        "created_at":        None,
        "updated_at":        None,
        "pushed_at":         None,
        "is_fork":           None,
        "default_branch":    None,
        "topics":            None,
        "has_wiki":          None,
        "has_discussions":   None,
        "contributor_count": None,
        "bus_factor":        None,
        "release_downloads": None,
        "top_languages":     None,
        # community / health
        "has_readme":              None,
        "has_contributing":        None,
        "has_code_of_conduct":     None,
        "has_license":             None,
        "has_security_policy":     None,
        "has_issue_template":      None,
        "has_pull_request_template": None,
        "health_percentage":       None,
    }

    if not repo_path:
        log.warning("  No github_repo for %s — skipping GitHub calls.", proj["name"])
        return row

    # ---- Core repo info ----
    repo = fetch_repo(repo_path)
    if repo:
        lic = repo.get("license") or {}
        license_val = lic.get("spdx_id") or lic.get("name")
        # GitHub can't always classify licenses (numpy, matplotlib, etc.).
        # If the API returns nothing, fall back to probing the repo root directly.
        if not license_val or license_val in ("NOASSERTION", ""):
            license_val = fetch_license_fallback(repo_path)
        row.update({
            "html_url":          repo.get("html_url"),
            "description":       repo.get("description"),
            "primary_language":  repo.get("language"),
            "license":           license_val,
            "stargazers_count":  repo.get("stargazers_count"),
            "forks_count":       repo.get("forks_count"),
            "open_issues_count": repo.get("open_issues_count"),
            "watchers_count":    repo.get("watchers_count"),
            "subscribers_count": repo.get("subscribers_count"),
            "size_kb":           repo.get("size"),
            "created_at":        repo.get("created_at"),
            "updated_at":        repo.get("updated_at"),
            "pushed_at":         repo.get("pushed_at"),
            "is_fork":           repo.get("fork"),
            "default_branch":    repo.get("default_branch"),
            "topics":            "|".join(repo.get("topics") or []),
            "has_wiki":          repo.get("has_wiki"),
            "has_discussions":   repo.get("has_discussions"),
        })
    else:
        log.warning("  Could not fetch repo info for %s", repo_path)

    # ---- Community health ----
    community = check_community_files(repo_path)
    # The community profile API also misses unclassified licenses.
    # If we already know a license file exists (from the fallback above),
    # make sure has_license reflects that — no extra API call needed.
    if not community.get("has_license") and row.get("license"):
        community["has_license"] = True
    row.update(community)

    # ---- Contributor count ----
    cc = fetch_contributor_count(repo_path)
    row["contributor_count"] = cc

    # ---- Bus factor proxy ----
    bf = bus_factor_estimate(repo_path)
    row["bus_factor"] = bf

    # ---- Release downloads ----
    rd = fetch_release_downloads(repo_path)
    row["release_downloads"] = rd

    # ---- Language breakdown (top 3) ----
    langs = fetch_languages(repo_path)
    if langs:
        sorted_langs = sorted(langs.items(), key=lambda x: x[1], reverse=True)
        row["top_languages"] = "|".join(l for l, _ in sorted_langs[:3])

    return row


def main():
    projects = json.loads(SEED_FILE.read_text(encoding="utf-8"))
    log.info("Loaded %d projects from seed file.", len(projects))

    rows = []
    for i, proj in enumerate(projects, 1):
        log.info("[%d/%d] %s", i, len(projects), proj["name"])
        try:
            row = collect_project(proj)
            rows.append(row)
        except Exception as exc:
            log.exception("Unexpected error for %s: %s", proj["name"], exc)
            rows.append({"name": proj["name"], "github_repo": proj.get("github_repo", "")})
        # Be polite to the API
        time.sleep(0.5)

    df = pd.DataFrame(rows)

    # ---- Dtype optimisation ----
    for col in ["stargazers_count", "forks_count", "open_issues_count",
                "watchers_count", "subscribers_count", "contributor_count",
                "bus_factor", "release_downloads", "sponsored_since",
                "health_percentage", "size_kb"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int32")

    for col in ["created_at", "updated_at", "pushed_at"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    df.to_parquet(OUT_PARQUET, index=False)
    log.info("Saved %d rows → %s", len(df), OUT_PARQUET)

    # ---- OpenSSF Scorecard ----
    log.info("Fetching OpenSSF Scorecard data …")
    sec_rows = []
    for i, proj in enumerate(projects, 1):
        repo_path = proj.get("github_repo", "")
        if not repo_path:
            continue
        log.info("[%d/%d] Scorecard  %s", i, len(projects), proj["name"])
        try:
            sc = fetch_scorecard(repo_path)
            if sc:
                sc["name"] = proj["name"]
                sec_rows.append(sc)
            else:
                sec_rows.append({
                    "github_repo": repo_path,
                    "name":        proj["name"],
                    "Total_Score": None,
                })
        except Exception as exc:
            log.exception("Scorecard error for %s: %s", proj["name"], exc)
        time.sleep(0.3)

    if sec_rows:
        df_sec = pd.DataFrame(sec_rows)
        # Ensure all check columns exist
        for check in SCORECARD_CHECKS:
            col = check.replace("-", "_")
            if col not in df_sec.columns:
                df_sec[col] = pd.NA
        df_sec.to_parquet(OUT_SECURITY_PARQUET, index=False)
        log.info("Saved %d scorecard rows → %s", len(df_sec), OUT_SECURITY_PARQUET)

    # ---- Quick summary ----
    print("\n=== Collection summary ===")
    print(f"Projects collected : {len(df)}")
    if "stargazers_count" in df.columns:
        print(f"Total stars        : {df['stargazers_count'].sum():,.0f}")
    if "contributor_count" in df.columns:
        print(f"Total contributors : {df['contributor_count'].sum():,.0f}")
    print(f"Output file        : {OUT_PARQUET}")


if __name__ == "__main__":
    main()
