#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# NumFocus Sponsored Projects Explorer
# A Shiny for Python app to browse, filter, and analyse the 63 NumFOCUS
# sponsored projects and their GitHub repository metrics.
# Run: python -m shiny run app.py

import os
import io
import logging
import pandas as pd
import altair as alt
from pathlib import Path

# ---------------------------------------------------------------------------
# Global Altair theme — larger axis/legend labels across all charts
# ---------------------------------------------------------------------------

def _numfocus_theme():
    return {
        "config": {
            "axis": {
                "labelFontSize": 13,
                "titleFontSize": 14,
                "titleFontWeight": "normal",
                "labelLimit": 200,
            },
            "legend": {
                "labelFontSize": 12,
                "titleFontSize": 13,
            },
            "header": {
                "labelFontSize": 13,
            },
            "view": {"stroke": None},
        }
    }

_ = alt.themes.register("numfocus", _numfocus_theme)
_ = alt.themes.enable("numfocus")
from shiny.express import input, ui, render
from shiny import reactive
from shiny import ui as sui
from shinywidgets import render_altair
from faicons import icon_svg

# ---------------------------------------------------------------------------
# Config & paths
# ---------------------------------------------------------------------------
PARQUET_PATH          = Path("Data/parquet/numfocus_projects.parquet")
SECURITY_PARQUET_PATH = Path("Data/parquet/numfocus_security.parquet")
SEED_PATH             = Path("Data/projects_seed.json")

NF_BLUE    = "#00A0DC"
NF_GREEN   = "#4DAF4A"
NF_RED     = "#E41A1C"
NF_ORANGE  = "#FF7F00"
NF_PURPLE  = "#984EA3"

ALL_LANGUAGES = ["Python", "R", "Julia", "JavaScript", "Multiple", "Other"]

ALL_FEATURES = [
    "Assistive Technology", "Browser Interactivity", "Big Data",
    "Computational thinking", "Computing Language", "Data Mining",
    "Data Wrangling", "Educational Outreach", "High Performance Computing",
    "Machine Learning", "Modeling", "Numerical Computing",
    "Statistical Computing", "Subject Area Libraries", "Text Processing",
    "Version control", "Visualization",
]

ALL_INDUSTRIES = [
    "Business & Industry Applications",
    "Higher Education Research & Teaching",
    "Government",
]

# OpenSSF Scorecard checks — (display label, parquet column name)
SCORECARD_CHECKS = [
    ("Binary Artifacts",      "Binary_Artifacts"),
    ("Branch Protection",     "Branch_Protection"),
    ("CI Best Practices",     "CI_Best_Practices"),
    ("Code Review",           "Code_Review"),
    ("Contributors",          "Contributors"),
    ("Dangerous Workflow",    "Dangerous_Workflow"),
    ("Dependency Update Tool","Dependency_Update_Tool"),
    ("Fuzzing",               "Fuzzing"),
    ("License",               "License"),
    ("Maintained",            "Maintained"),
    ("Packaging",             "Packaging"),
    ("Pinned Dependencies",   "Pinned_Dependencies"),
    ("SAST",                  "SAST"),
    ("Security Policy",       "Security_Policy"),
    ("Signed Releases",       "Signed_Releases"),
    ("Token Permissions",     "Token_Permissions"),
    ("Vulnerabilities",       "Vulnerabilities"),
    ("Total Score",           "Total_Score"),
]

FEATURES_HEALTH = [
    "has_readme", "has_contributing", "has_code_of_conduct",
    "has_license", "has_security_policy", "has_issue_template",
    "has_pull_request_template",
]

FEATURE_LABELS = {
    "has_readme":                "README",
    "has_contributing":          "Contributing Guide",
    "has_code_of_conduct":       "Code of Conduct",
    "has_license":               "License",
    "has_security_policy":       "Security Policy",
    "has_issue_template":        "Issue Templates",
    "has_pull_request_template": "PR Template",
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_data() -> pd.DataFrame:
    if PARQUET_PATH.exists():
        df = pd.read_parquet(PARQUET_PATH)
        logging.info("Loaded %d projects from parquet.", len(df))
        return df
    if SEED_PATH.exists():
        import json
        logging.warning("Parquet not found — loading seed JSON (no GitHub metrics yet).")
        records = json.loads(SEED_PATH.read_text(encoding="utf-8"))
        df = pd.DataFrame(records)
        for col in ["stargazers_count", "forks_count", "open_issues_count",
                    "contributor_count", "bus_factor", "release_downloads",
                    "watchers_count", "health_percentage"]:
            if col not in df.columns:
                df[col] = pd.NA
        return df
    return pd.DataFrame()


df_raw = _load_data()

# Load security / scorecard data (optional — only exists after collection)
if SECURITY_PARQUET_PATH.exists():
    df_security = pd.read_parquet(SECURITY_PARQUET_PATH)
    logging.info("Loaded scorecard data for %d projects.", len(df_security))
else:
    df_security = pd.DataFrame()


def _preprocess(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    for col in ["language_tags", "feature_tags", "industry_tags"]:
        if col not in df.columns:
            df[col] = ""
    for col in ["stargazers_count", "forks_count", "open_issues_count",
                "watchers_count", "contributor_count", "bus_factor",
                "release_downloads", "health_percentage", "sponsored_since"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in FEATURES_HEALTH:
        if col in df.columns:
            df[col] = df[col].map(
                lambda v: True  if v is True  or str(v).lower() in ("true",  "1", "yes")
                else (False if v is False or str(v).lower() in ("false", "0", "no")
                else pd.NA)
            )
    if "license" in df.columns:
        df["license"] = df["license"].astype("string").str.strip()
        df["license"] = df["license"].mask(
            df["license"].str.lower().isin(["", "none", "nan", "null", "<na>", "noassertion"]),
            pd.NA,
        )
    return df


df = _preprocess(df_raw.copy())

# Unique values for filter widgets
def _unique_tags(col):
    if df.empty or col not in df.columns:
        return []
    tags = set()
    for v in df[col].dropna():
        for t in str(v).split("|"):
            if t.strip():
                tags.add(t.strip())
    return sorted(tags)

_lang_vals  = _unique_tags("language_tags")  or ALL_LANGUAGES
_feat_vals  = _unique_tags("feature_tags")   or ALL_FEATURES
_ind_vals   = _unique_tags("industry_tags")  or ALL_INDUSTRIES
_licenses   = sorted(df["license"].dropna().unique().tolist()) if "license" in df.columns else []
_stars_max  = int(pd.to_numeric(df.get("stargazers_count"), errors="coerce").max() or 1000)
_forks_max  = int(pd.to_numeric(df.get("forks_count"),     errors="coerce").max() or 500)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe(v, default="—"):
    if v is None:
        return default
    try:
        if pd.isna(v):
            return default
    except Exception:
        pass
    return str(v)

def _fmt_int(v):
    try:
        if pd.isna(v):
            return "N/A"
        return f"{int(float(v)):,}"
    except Exception:
        return "N/A"

def _pct(part, whole):
    return "0%" if whole == 0 else f"{100 * part / whole:.1f}%"

def _bool_icon(v):
    if v is True  or str(v).lower() in ("true",  "1"): return "✅"
    if v is False or str(v).lower() in ("false", "0"): return "✗"
    return "—"

def _tag_counts(data: pd.DataFrame, col: str) -> pd.DataFrame:
    """Explode a pipe-separated tag column into a count DataFrame."""
    counts: dict = {}
    for v in data[col].dropna():
        for tag in str(v).split("|"):
            tag = tag.strip()
            if tag:
                counts[tag] = counts.get(tag, 0) + 1
    if not counts:
        return pd.DataFrame(columns=["tag", "count"])
    return (
        pd.DataFrame(list(counts.items()), columns=["tag", "count"])
        .sort_values("count", ascending=False)
    )

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

ui.tags.style("""
.nf-header-bar {
    background: linear-gradient(90deg, #00A0DC 0%, #005A8B 100%);
    color: white;
    padding: 8px 16px;
    border-radius: 6px;
    margin-bottom: 10px;
    font-size: 1.1rem;
    font-weight: 600;
}
.nav-pills .nav-link, .nav-tabs .nav-link { font-size: 0.9rem; }
.bslib-sidebar .form-label,
.bslib-sidebar .control-label,
aside .control-label,
aside .form-label { font-size: 0.82rem !important; }
.bslib-sidebar, aside[data-bslib-sidebar], [data-bslib-sidebar] {
    resize: horizontal; overflow: auto; min-width: 200px; max-width: 60%;
}
.metric-label { color: #00A0DC; font-weight: bold; }
.tag-chip {
    display: inline-block;
    background: #e8f4fd; color: #005A8B;
    border: 1px solid #00A0DC; border-radius: 12px;
    padding: 2px 8px; font-size: 0.75rem; margin: 2px;
}
.data-missing-banner {
    background: #fff3cd; border: 1px solid #ffc107;
    border-radius: 6px; padding: 10px 16px;
    margin-bottom: 12px; font-size: 0.88rem;
}
""")

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

ui.page_opts(title="NumFOCUS Project Explorer", fillable=True)

_data_ready = PARQUET_PATH.exists()

with ui.sidebar(open="open", bg="#f4f8fc", width="300px"):
    with ui.navset_pill(id="side_tab"):
        with ui.nav_panel("Filters"):
            if not _data_ready:
                ui.HTML("""
                <div class="data-missing-banner">
                    ⚠️ GitHub metrics not yet collected.<br>
                    Run <code>bash run_collection.sh</code>, then restart.
                </div>
                """)
            ui.input_selectize("language", "Language:",
                               choices=_lang_vals, multiple=True)
            ui.input_selectize("feature",  "Features:",
                               choices=_feat_vals, multiple=True)
            ui.input_selectize("industry", "Industry:",
                               choices=_ind_vals,  multiple=True)
            ui.input_selectize("license",  "License:",
                               choices=_licenses,  multiple=True)
            ui.input_slider("slider_stars", "Min ⭐ Stars",
                            min=0, max=_stars_max, value=0, step=100)
            ui.input_slider("slider_forks", "Min Forks",
                            min=0, max=_forks_max, value=0, step=10)
            ui.input_slider("slider_since", "Sponsored Since (year)",
                            min=2012, max=2025, value=2012, step=1)
    ui.br()
    ui.input_action_button("reset_filters", "Reset All Filters",
                           class_="btn-danger btn-sm")
    ui.HTML("")


@reactive.effect
@reactive.event(input.reset_filters)
def _reset():
    ui.update_selectize("language", selected=[])
    ui.update_selectize("feature",  selected=[])
    ui.update_selectize("industry", selected=[])
    ui.update_selectize("license",  selected=[])
    ui.update_slider("slider_stars", value=0)
    ui.update_slider("slider_forks", value=0)
    ui.update_slider("slider_since", value=2012)
    ui.update_text("table_search", value="")


ICONS = {
    "projects":     icon_svg("diagram-project"),
    "stars":        icon_svg("star"),
    "contributors": icon_svg("users"),
    "license":      icon_svg("id-card"),
    "forks":        icon_svg("code-fork"),
    "downloads":    icon_svg("download"),
    "bus":          icon_svg("bus"),
    "health":       icon_svg("heart-pulse"),
}

# ---------------------------------------------------------------------------
# Main tabs
# ---------------------------------------------------------------------------

with ui.navset_pill(id="main_tab", selected="Overview"):

    # ── ABOUT ──────────────────────────────────────────────────────────────
    with ui.nav_panel("About"):
        with ui.card():
            ui.HTML("""
            <div class="nf-header-bar">🔬 NumFOCUS Sponsored Projects Explorer</div>
            <p>
              Explore all <strong>63 NumFOCUS fiscally-sponsored projects</strong>
              — from NumPy and pandas to Zarr, Parsl, and OpenProblems.bio —
              through their GitHub metrics, community health indicators, and domain taxonomy.
            </p>

            <h4>What's in the data?</h4>
            <ul>
              <li><strong>GitHub metrics</strong> — stars, forks, open issues, contributors,
                  release downloads, bus factor (top-contributor concentration estimate)</li>
              <li><strong>Community health</strong> — README, contributing guide, code of conduct,
                  security policy, issue templates, PR template</li>
              <li><strong>NumFOCUS taxonomy</strong> — programming language, feature area, and
                  target industry, as tagged on
                  <a href="https://numfocus.org/sponsored-projects" target="_blank">numfocus.org</a></li>
            </ul>

            <h4>How is the health score computed?</h4>
            <p>
              The <em>GitHub health percentage</em> is returned directly by GitHub's
              <code>/repos/{owner}/{repo}/community/profile</code> API endpoint.
              GitHub calculates it by checking the presence of six community files:
              README, LICENSE, CONTRIBUTING, CODE_OF_CONDUCT, ISSUE_TEMPLATE, and
              PULL_REQUEST_TEMPLATE. Each file that exists contributes to the score
              (GitHub does not publish the exact weights). A score of 100 means all
              six files are present. We additionally store individual boolean flags for
              each file so you can see exactly which ones are missing per project.
            </p>
            <p>
              The <em>bus factor estimate</em> is a rough proxy: it counts how many
              top contributors are needed to account for at least 50% of total commits,
              based on GitHub's contributor list (capped at 100 contributors).
            </p>

            <h4>How to use</h4>
            <p>Use the sidebar to narrow projects by language, feature area, industry,
            license, stars, forks, or sponsorship year. Switch between the
            <em>Overview</em>, <em>Projects</em>, <em>Impact</em>, and
            <em>Sustainability</em> tabs. Click any row in the Projects table
            for a detailed panel.</p>

            <h4>Refreshing data</h4>
            <p>Run <code>bash run_collection.sh</code> from the project folder, then
            restart the app.</p>
            <hr>
            <p style="color:#888; font-size:0.85rem;">
              Built with <a href="https://shiny.posit.co/py/" target="_blank">Shiny for Python</a>
              and <a href="https://altair-viz.github.io/" target="_blank">Altair</a> •
              Data from <a href="https://numfocus.org" target="_blank">NumFOCUS</a>
              and the <a href="https://docs.github.com/en/rest" target="_blank">GitHub API</a>
            </p>
            """)

    # ── OVERVIEW ────────────────────────────────────────────────────────────
    with ui.nav_panel("Overview"):

        with ui.layout_columns(col_widths=(3, 3, 3, 3)):
            with ui.value_box(showcase=ICONS["projects"]):
                "Sponsored projects"
                @render.express
                def vb_total():
                    len(filtered_df())

            with ui.value_box(showcase=ICONS["stars"]):
                "Total ⭐ stars"
                @render.express
                def vb_stars():
                    s = pd.to_numeric(filtered_df().get("stargazers_count",
                                      pd.Series(dtype=float)), errors="coerce")
                    f"{int(s.dropna().sum()):,}" if s.notna().any() else "—"

            with ui.value_box(showcase=ICONS["contributors"]):
                "Total contributors"
                @render.express
                def vb_contributors():
                    s = pd.to_numeric(filtered_df().get("contributor_count",
                                      pd.Series(dtype=float)), errors="coerce")
                    v = int(s.dropna().sum()) if s.notna().any() else 0
                    f"{v:,}" if v > 0 else "—"

            with ui.value_box(showcase=ICONS["license"]):
                "Projects with a license"
                @render.express
                def vb_license():
                    data = filtered_df()
                    if "license" not in data.columns or data.empty:
                        "—"
                    else:
                        n = int(data["license"].notna().sum())
                        _pct(n, len(data))

        with ui.layout_columns(col_widths=(5, 7)):
            with ui.card(full_screen=True):
                ui.card_header("Programming Language")
                @render_altair
                def plot_language():
                    return _altair_hbar(
                        filtered_df(), "language_tags", "Language", NF_BLUE)

            with ui.card(full_screen=True):
                ui.card_header("Feature Area Distribution")
                @render_altair
                def plot_features():
                    return _altair_hbar(
                        filtered_df(), "feature_tags", "Feature", NF_GREEN,
                        max_items=17, chart_height=340)

        with ui.layout_columns(col_widths=(5, 7)):
            with ui.card(full_screen=True):
                ui.card_header("Target Industry")
                @render_altair
                def plot_industry():
                    return _altair_hbar(
                        filtered_df(), "industry_tags", "Industry", NF_RED,
                        chart_height=160)

            with ui.card(full_screen=True):
                ui.card_header("Projects Joining NumFOCUS per Year")
                @render_altair
                def plot_sponsorship_timeline():
                    data = filtered_df()
                    if "sponsored_since" not in data.columns or data.empty:
                        return alt.Chart(pd.DataFrame()).mark_text().encode(
                            text=alt.value("No data"))
                    counts = (
                        pd.to_numeric(data["sponsored_since"], errors="coerce")
                        .dropna().astype(int)
                        .value_counts().reset_index()
                    )
                    counts.columns = ["year", "count"]
                    counts["year"] = counts["year"].astype(str)
                    return (
                        alt.Chart(counts)
                        .mark_bar(color=NF_BLUE, cornerRadiusTopLeft=3,
                                  cornerRadiusTopRight=3)
                        .encode(
                            x=alt.X("year:O", title="Year",
                                    axis=alt.Axis(labelAngle=-45)),
                            y=alt.Y("count:Q", title="Projects"),
                            tooltip=[
                                alt.Tooltip("year:O", title="Year"),
                                alt.Tooltip("count:Q", title="Projects"),
                            ],
                        )
                        .properties(height=200)
                    )

        with ui.card(full_screen=True):
            ui.card_header(
                "Community Health — % of Projects with Each File  "
                "| 🟢 ≥75%  🟠 40–74%  🔴 <40%"
            )
            @render_altair
            def plot_health_overview():
                return _altair_health_bar(filtered_df())

    # ── PROJECTS ─────────────────────────────────────────────────────────────
    with ui.nav_panel("Projects"):
        with ui.card():
            ui.input_text("table_search", "Search",
                          placeholder="Search by name, description, language…",
                          width="100%")
            @render.data_frame
            def projects_table():
                return render.DataGrid(
                    _projects_table_df(), height="450px", selection_mode="row")

            @render.download(
                filename=lambda: f"numfocus_projects_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
            def download_csv():
                buf = io.BytesIO()
                _projects_table_df().to_csv(buf, index=False, encoding="utf-8")
                buf.seek(0)
                yield buf.getvalue()

        with ui.card():
            @render.ui
            def project_detail():
                sel_rows = projects_table.cell_selection()["rows"]
                if not sel_rows:
                    return ui.p("Click a row above to see project details.",
                                class_="text-muted")

                view = _projects_table_df()
                sel  = filtered_df().loc[view.index[sel_rows[0]]]
                if isinstance(sel, pd.DataFrame):
                    sel = sel.iloc[0]

                def _chips(col):
                    raw = _safe(sel.get(col), "")
                    if not raw or raw == "—":
                        return ui.span("—")
                    return ui.div(*[
                        ui.span(t.strip(), class_="tag-chip")
                        for t in raw.split("|") if t.strip()
                    ])

                gh_url   = _safe(sel.get("html_url"),      "")
                nf_url   = _safe(sel.get("numfocus_url"),  "")
                website  = _safe(sel.get("website"),        "")
                is_org   = bool(sel.get("is_org", False))
                gh_org   = _safe(sel.get("github_org"),    "")
                pub_repos = sel.get("public_repos")

                def _link(url):
                    return (ui.tags.a(url, href=url, target="_blank")
                            if url and url != "—" else ui.span("—"))

                return sui.layout_columns(
                    ui.div(
                        ui.div(
                            ui.h4(_safe(sel.get("name"), "Unknown"),
                                  style="color:#00A0DC; margin-bottom:6px;"),
                            style="border-left:4px solid #00A0DC; padding-left:12px;",
                        ),
                        sui.navset_tab(
                            sui.nav_panel("Overview",
                                ui.p(ui.tags.span("Description: ", class_="metric-label"),
                                     _safe(sel.get("description"))),
                                ui.p(ui.tags.span("Language tags: ",  class_="metric-label"),
                                     _chips("language_tags")),
                                ui.p(ui.tags.span("Feature tags: ",   class_="metric-label"),
                                     _chips("feature_tags")),
                                ui.p(ui.tags.span("Industry: ",       class_="metric-label"),
                                     _chips("industry_tags")),
                                ui.p(ui.tags.span("GitHub language: ",class_="metric-label"),
                                     _safe(sel.get("primary_language"))),
                                ui.p(ui.tags.span("License: ",        class_="metric-label"),
                                     _safe(sel.get("license"))),
                                ui.p(ui.tags.span("Sponsored since: ",class_="metric-label"),
                                     _safe(sel.get("sponsored_since"))),
                                *([ ui.p(ui.tags.span("Type: ", class_="metric-label"),
                                         ui.span("GitHub Organisation 🏢",
                                                 style="color:#888; font-style:italic;")),
                                    ui.p(ui.tags.span("Public repos: ", class_="metric-label"),
                                         str(int(pub_repos)) if pub_repos is not None and pd.notna(pub_repos) else "—"),
                                  ] if is_org else []),
                                ui.p(ui.tags.span("GitHub: ",         class_="metric-label"),
                                     _link(gh_url)),
                                ui.p(ui.tags.span("NumFOCUS page: ",  class_="metric-label"),
                                     _link(nf_url)),
                                ui.p(ui.tags.span("Website: ",        class_="metric-label"),
                                     _link(website)),
                            ),
                            sui.nav_panel("Impact",
                                ui.p("⚠️ Stats below are aggregated from this organisation's top-10 repos by stars.",
                                     class_="text-muted small",
                                     style="font-style:italic; margin-bottom:8px;",
                                ) if is_org else ui.span(""),
                                ui.tags.table(
                                    *[ui.tags.tr(
                                        ui.tags.td(lbl, class_="metric-label",
                                                   style="padding-right:12px; padding-bottom:6px;"),
                                        ui.tags.td(_fmt_int(sel.get(col))),
                                      ) for lbl, col in [
                                        ("⭐ Stars",            "stargazers_count"),
                                        ("🍴 Forks",            "forks_count"),
                                        ("🐛 Open issues",      "open_issues_count"),
                                        ("👀 Watchers",         "watchers_count"),
                                        ("👥 Contributors",     "contributor_count"),
                                        ("🚌 Bus factor (est.)","bus_factor"),
                                        ("⬇️ Release downloads","release_downloads"),
                                    ]],
                                    style="width:100%; border-collapse:collapse;",
                                ),
                            ),
                            sui.nav_panel("Health",
                                ui.tags.table(
                                    ui.tags.tr(
                                        ui.tags.th("File / Policy",
                                                   style="text-align:left; padding-bottom:6px;"),
                                        ui.tags.th("Present",
                                                   style="text-align:center; padding-bottom:6px;"),
                                    ),
                                    *[ui.tags.tr(
                                        ui.tags.td(FEATURE_LABELS.get(col, col),
                                                   class_="metric-label",
                                                   style="padding-right:12px; padding-bottom:4px;"),
                                        ui.tags.td(_bool_icon(sel.get(col)),
                                                   style="text-align:center;"),
                                      ) for col in FEATURES_HEALTH],
                                    ui.tags.tr(
                                        ui.tags.td("Health % (GitHub)", class_="metric-label",
                                                   style="padding-right:12px; padding-top:8px;"),
                                        ui.tags.td(
                                            (_safe(sel.get("health_percentage"), "—") + "%"
                                             if _safe(sel.get("health_percentage"), "—") != "—"
                                             else "—"),
                                            style="text-align:center; padding-top:8px;",
                                        ),
                                    ),
                                    style="width:100%; border-collapse:collapse;",
                                ),
                            ),
                            id="proj_detail_tabs",
                        ),
                        style="padding-right:16px; border-right:1px solid #ddd;",
                    ),
                    ui.div(
                        ui.h5("Language Breakdown (GitHub)", style="color:#555;"),
                        _lang_spark_ui(sel),
                        style="padding-left:16px;",
                    ),
                    col_widths=(7, 5),
                )

    # ── IMPACT ───────────────────────────────────────────────────────────────
    with ui.nav_panel("Impact"):
        with ui.layout_columns(col_widths=(3, 3, 3, 3)):
            with ui.value_box(showcase=ICONS["stars"]):
                "Total ⭐ stars"
                @render.express
                def imp_stars():
                    s = pd.to_numeric(filtered_df().get("stargazers_count",
                                      pd.Series(dtype=float)), errors="coerce")
                    f"{int(s.dropna().sum()):,}" if s.notna().any() else "—"

            with ui.value_box(showcase=ICONS["forks"]):
                "Total forks"
                @render.express
                def imp_forks():
                    s = pd.to_numeric(filtered_df().get("forks_count",
                                      pd.Series(dtype=float)), errors="coerce")
                    f"{int(s.dropna().sum()):,}" if s.notna().any() else "—"

            with ui.value_box(showcase=ICONS["contributors"]):
                "Total contributors"
                @render.express
                def imp_contributors():
                    s = pd.to_numeric(filtered_df().get("contributor_count",
                                      pd.Series(dtype=float)), errors="coerce")
                    f"{int(s.dropna().sum()):,}" if s.notna().any() else "—"

            with ui.value_box(showcase=ICONS["downloads"]):
                "Total release downloads"
                @render.express
                def imp_downloads():
                    s = pd.to_numeric(filtered_df().get("release_downloads",
                                      pd.Series(dtype=float)), errors="coerce")
                    f"{int(s.dropna().sum()):,}" if s.notna().any() else "—"

        with ui.layout_columns(col_widths=(6, 6)):
            with ui.card(full_screen=True):
                ui.card_header("Top Projects by ⭐ Stars")
                @render_altair
                def plot_stars_bar():
                    return _altair_top_n(filtered_df(), "stargazers_count",
                                         "Stars", NF_BLUE)

            with ui.card(full_screen=True):
                ui.card_header("Top Projects by 👥 Contributors")
                @render_altair
                def plot_contrib_bar():
                    return _altair_top_n(filtered_df(), "contributor_count",
                                         "Contributors", NF_GREEN)

        with ui.layout_columns(col_widths=(6, 6)):
            with ui.card(full_screen=True):
                ui.card_header("Top Projects by 🍴 Forks")
                @render_altair
                def plot_forks_bar():
                    return _altair_top_n(filtered_df(), "forks_count",
                                         "Forks", NF_ORANGE)

            with ui.card(full_screen=True):
                ui.card_header("Top Projects by ⬇️ Release Downloads")
                @render_altair
                def plot_downloads_bar():
                    return _altair_top_n(filtered_df(), "release_downloads",
                                         "Downloads", NF_PURPLE)

        with ui.card(full_screen=True):
            ui.card_header("Impact Leaderboard")
            @render.data_frame
            def impact_leaderboard():
                data = filtered_df()
                if data.empty:
                    return render.DataGrid(pd.DataFrame())
                col_map = {
                    "name":              "Project",
                    "language_tags":     "Language",
                    "stargazers_count":  "Stars",
                    "forks_count":       "Forks",
                    "contributor_count": "Contributors",
                    "release_downloads": "Downloads",
                    "sponsored_since":   "Since",
                }
                out = data[[c for c in col_map if c in data.columns]].rename(columns=col_map)
                for col in ["Stars", "Forks", "Contributors", "Downloads"]:
                    if col in out.columns:
                        out[col] = pd.to_numeric(out[col], errors="coerce")
                if "Stars" in out.columns:
                    out = out.sort_values("Stars", ascending=False, na_position="last")
                for col in ["Stars", "Forks", "Contributors", "Downloads"]:
                    if col in out.columns:
                        out[col] = out[col].map(
                            lambda v: f"{int(v):,}" if pd.notna(v) else "—")
                return render.DataGrid(out, width="100%", height="400px")

    # ── SUSTAINABILITY ────────────────────────────────────────────────────────
    with ui.nav_panel("Sustainability"):
        with ui.layout_columns(col_widths=(6, 6)):
            with ui.value_box(showcase=ICONS["bus"]):
                "Avg bus factor"
                @render.express
                def sus_bus():
                    s = pd.to_numeric(filtered_df().get("bus_factor",
                                      pd.Series(dtype=float)), errors="coerce")
                    f"{s.mean():.1f}" if s.notna().any() else "—"

            with ui.value_box(showcase=ICONS["health"]):
                "Avg GitHub health %"
                @render.express
                def sus_health():
                    s = pd.to_numeric(filtered_df().get("health_percentage",
                                      pd.Series(dtype=float)), errors="coerce")
                    f"{s.mean():.0f}%" if s.notna().any() else "—"

        with ui.layout_columns(col_widths=(7, 5)):
            with ui.card(full_screen=True):
                ui.card_header("Community Health Files — Per Project Heatmap")
                @render_altair
                def plot_health_heatmap():
                    return _altair_health_heatmap(filtered_df())

            with ui.card(full_screen=True):
                ui.card_header("Bus Factor Distribution")
                @render_altair
                def plot_bus_factor():
                    data = filtered_df()
                    if data.empty or "bus_factor" not in data.columns:
                        return _empty_chart("No data")
                    s = pd.to_numeric(data["bus_factor"], errors="coerce").dropna()
                    if s.empty:
                        return _empty_chart("Run data collection first")
                    counts = (
                        s.astype(int).value_counts().reset_index()
                    )
                    counts.columns = ["bus_factor", "count"]
                    counts["bus_factor"] = counts["bus_factor"].astype(str)
                    return (
                        alt.Chart(counts)
                        .mark_bar(color=NF_BLUE, cornerRadiusTopLeft=3,
                                  cornerRadiusTopRight=3)
                        .encode(
                            x=alt.X("bus_factor:O",
                                    title="Bus Factor (estimate)",
                                    sort=alt.EncodingSortField(
                                        field="bus_factor", order="ascending"),
                                    axis=alt.Axis(labelAngle=0)),
                            y=alt.Y("count:Q", title="Projects"),
                            tooltip=[
                                alt.Tooltip("bus_factor:O", title="Bus Factor"),
                                alt.Tooltip("count:Q", title="Projects"),
                            ],
                        )
                        .properties(height=280)
                    )

        with ui.card(full_screen=True):
            ui.card_header(
                "Health Score vs ⭐ Stars  "
                "— GitHub health% (y) against star count (x), coloured by language"
            )
            @render_altair
            def plot_health_vs_stars():
                return _altair_health_vs_stars(filtered_df())

        with ui.card(full_screen=True):
            ui.card_header("Sustainability Indicators per Feature Area")
            @render.data_frame
            def sustainability_table():
                data = filtered_df()
                if data.empty or "feature_tags" not in data.columns:
                    return render.DataGrid(pd.DataFrame())
                expanded = []
                for _, row in data.iterrows():
                    for tag in str(row.get("feature_tags", "")).split("|"):
                        tag = tag.strip()
                        if tag:
                            expanded.append({**row.to_dict(), "_feature": tag})
                if not expanded:
                    return render.DataGrid(pd.DataFrame())
                exp = pd.DataFrame(expanded)
                rows = []
                for feat, grp in exp.groupby("_feature"):
                    cc = pd.to_numeric(grp.get("contributor_count",
                                               pd.Series(dtype=float)), errors="coerce")
                    bf = pd.to_numeric(grp.get("bus_factor",
                                               pd.Series(dtype=float)), errors="coerce")
                    hp = pd.to_numeric(grp.get("health_percentage",
                                               pd.Series(dtype=float)), errors="coerce")
                    rows.append({
                        "Feature Area":      feat,
                        "Projects":          int(grp["name"].nunique()) if "name" in grp.columns else len(grp),
                        "Avg Contributors":  f"{cc.mean():.1f}" if cc.notna().any() else "—",
                        "Avg Bus Factor":    f"{bf.mean():.1f}" if bf.notna().any() else "—",
                        "Avg Health %":      f"{hp.mean():.0f}%" if hp.notna().any() else "—",
                    })
                out = pd.DataFrame(rows).sort_values("Projects", ascending=False)
                return render.DataGrid(out, width="100%")

    # ── SECURITY ─────────────────────────────────────────────────────────────
    with ui.nav_panel("Security"):
        if df_security.empty:
            with ui.card():
                ui.HTML("""
                <div class="data-missing-banner">
                    ⚠️ No scorecard data yet. Re-run <code>bash run_collection.sh</code>
                    to fetch OpenSSF Scorecard results, then restart the app.
                </div>
                """)
        else:
            with ui.layout_columns(col_widths=(3, 3, 3, 3)):
                with ui.value_box(showcase=icon_svg("shield-halved")):
                    "Avg total score (0–10)"
                    @render.express
                    def sec_avg_score():
                        data = _security_table_df()
                        s = pd.to_numeric(data.get("Total Score",
                                          pd.Series(dtype=float)), errors="coerce")
                        f"{s.mean():.2f}" if s.notna().any() else "—"

                with ui.value_box(showcase=icon_svg("shield")):
                    "Projects scored"
                    @render.express
                    def sec_n_scored():
                        data = _security_table_df()
                        s = pd.to_numeric(data.get("Total Score",
                                          pd.Series(dtype=float)), errors="coerce")
                        f"{s.notna().sum()} / {len(data)}"

                with ui.value_box(showcase=icon_svg("check")):
                    "Score ≥ 5  (passing)"
                    @render.express
                    def sec_n_passing():
                        data = _security_table_df()
                        s = pd.to_numeric(data.get("Total Score",
                                          pd.Series(dtype=float)), errors="coerce")
                        n = int((s >= 5).sum())
                        f"{n} / {s.notna().sum()}"

                with ui.value_box(showcase=icon_svg("triangle-exclamation")):
                    "Score < 5  (needs work)"
                    @render.express
                    def sec_n_failing():
                        data = _security_table_df()
                        s = pd.to_numeric(data.get("Total Score",
                                          pd.Series(dtype=float)), errors="coerce")
                        n = int((s < 5).sum())
                        f"{n} / {s.notna().sum()}"

            with ui.layout_columns(col_widths=(8, 4)):
                with ui.card(full_screen=True):
                    ui.card_header(
                        "Scorecard by project  "
                        "([OpenSSF Scorecard](https://scorecard.dev/))"
                    )
                    @render.data_frame
                    def security_scorecard_table():
                        out = _security_table_df()
                        if out.empty:
                            return render.DataGrid(out)
                        return render.DataGrid(
                            out,
                            width="100%",
                            height="600px",
                            styles=[{
                                "location": "body",
                                "style": {"fontSize": "12px"},
                            }],
                        )

                with ui.card(full_screen=True):
                    ui.card_header("Average score per check")
                    @render_altair
                    def plot_scorecard_averages():
                        return _altair_scorecard_averages(_security_table_df())


# ---------------------------------------------------------------------------
# Reactive data
# ---------------------------------------------------------------------------

@reactive.calc
def filtered_df() -> pd.DataFrame:
    if df.empty:
        return df
    mask = pd.Series(True, index=df.index)

    if input.language():
        sel = set(input.language())
        mask &= df["language_tags"].apply(
            lambda v: bool(sel & set(str(v).split("|"))) if pd.notna(v) else False)
    if input.feature():
        sel = set(input.feature())
        mask &= df["feature_tags"].apply(
            lambda v: bool(sel & set(str(v).split("|"))) if pd.notna(v) else False)
    if input.industry():
        sel = set(input.industry())
        mask &= df["industry_tags"].apply(
            lambda v: bool(sel & set(str(v).split("|"))) if pd.notna(v) else False)
    if input.license() and "license" in df.columns:
        mask &= df["license"].isin(input.license())
    if "stargazers_count" in df.columns:
        stars = pd.to_numeric(df["stargazers_count"], errors="coerce")
        mask &= (stars >= input.slider_stars()) | stars.isna()
    if "forks_count" in df.columns:
        forks = pd.to_numeric(df["forks_count"], errors="coerce")
        mask &= (forks >= input.slider_forks()) | forks.isna()
    if "sponsored_since" in df.columns:
        yr = pd.to_numeric(df["sponsored_since"], errors="coerce")
        mask &= (yr >= input.slider_since()) | yr.isna()

    return df.loc[mask]


@reactive.calc
def _projects_table_df() -> pd.DataFrame:
    data = filtered_df()
    show = [c for c in ["name", "is_org", "language_tags", "feature_tags",
                         "primary_language", "license", "stargazers_count",
                         "forks_count", "contributor_count",
                         "sponsored_since", "html_url"] if c in data.columns]
    out = data[show].copy()

    # Prefix org entries with a badge so they're visually distinct in the table
    if "is_org" in out.columns and "name" in out.columns:
        out["name"] = out.apply(
            lambda r: f"🏢 {r['name']}" if r.get("is_org") else r["name"], axis=1
        )
        out = out.drop(columns=["is_org"])

    out = out.rename(columns={
        "name": "Project", "language_tags": "Language",
        "feature_tags": "Features", "primary_language": "GitHub lang",
        "license": "License", "stargazers_count": "Stars",
        "forks_count": "Forks", "contributor_count": "Contributors",
        "sponsored_since": "Since", "html_url": "GitHub URL",
    })
    q = (input.table_search() or "").strip().lower()
    if q:
        src = filtered_df()
        m = pd.Series(False, index=src.index)
        for col in ["name", "description", "language_tags",
                    "feature_tags", "primary_language", "license"]:
            if col in src.columns:
                m |= src[col].astype(str).str.lower().str.contains(q, na=False)
        out = out[m]
    return out


# ---------------------------------------------------------------------------
# Altair chart helpers
# ---------------------------------------------------------------------------

def _empty_chart(msg: str = "No data available") -> alt.Chart:
    return (
        alt.Chart(pd.DataFrame({"x": [0.5], "y": [0.5], "text": [msg]}))
        .mark_text(fontSize=14, color="#888")
        .encode(
            x=alt.X("x:Q", axis=None),
            y=alt.Y("y:Q", axis=None),
            text="text:N",
        )
        .properties(width="container", height=200)
    )


def _altair_hbar(
    data: pd.DataFrame,
    col: str,
    tag_label: str,
    color: str = NF_BLUE,
    max_items: int = 20,
    chart_height: int = 280,
) -> alt.Chart:
    counts_df = _tag_counts(data, col).head(max_items)
    if counts_df.empty:
        return _empty_chart()
    counts_df = counts_df.rename(columns={"tag": tag_label, "count": "Projects"})
    return (
        alt.Chart(counts_df)
        .mark_bar(color=color, cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
        .encode(
            x=alt.X("Projects:Q", title="Number of projects"),
            y=alt.Y(f"{tag_label}:N", sort="-x", title=""),
            tooltip=[
                alt.Tooltip(f"{tag_label}:N", title=tag_label),
                alt.Tooltip("Projects:Q", title="Projects"),
            ],
        )
        .properties(height=chart_height, width="container")
    )


def _altair_health_bar(data: pd.DataFrame) -> alt.Chart:
    if data.empty:
        return _empty_chart()
    health_cols = [c for c in FEATURES_HEALTH if c in data.columns]
    if not health_cols:
        return _empty_chart("Run data collection first")

    n = len(data)
    rows = []
    for col in health_cols:
        have = data[col].map(
            lambda v: v is True or str(v).lower() in ("true", "1")
        ).sum()
        pct = round(100 * have / n, 1) if n > 0 else 0.0
        rows.append({
            "File": FEATURE_LABELS.get(col, col),
            "Pct": pct,
            "color_group": "≥75%" if pct >= 75 else ("40–74%" if pct >= 40 else "<40%"),
        })

    df_h = pd.DataFrame(rows)
    color_scale = alt.Scale(
        domain=["≥75%", "40–74%", "<40%"],
        range=[NF_GREEN, NF_ORANGE, NF_RED],
    )
    return (
        alt.Chart(df_h)
        .mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
        .encode(
            x=alt.X("Pct:Q", title="% of projects", scale=alt.Scale(domain=[0, 110])),
            y=alt.Y("File:N", sort="-x", title=""),
            color=alt.Color("color_group:N",
                            scale=color_scale,
                            legend=alt.Legend(title="Threshold")),
            tooltip=[
                alt.Tooltip("File:N",         title="File"),
                alt.Tooltip("Pct:Q",          title="% of projects", format=".1f"),
                alt.Tooltip("color_group:N",  title="Category"),
            ],
        )
        .properties(height=200, width="container")
    )


def _altair_top_n(
    data: pd.DataFrame,
    col: str,
    label: str,
    color: str,
    n: int = 20,
) -> alt.Chart:
    if data.empty or col not in data.columns:
        return _empty_chart()
    name_col = "name" if "name" in data.columns else data.columns[0]
    plot_df = (
        data[[name_col, col]]
        .assign(**{col: pd.to_numeric(data[col], errors="coerce")})
        .dropna(subset=[col])
        .sort_values(col, ascending=False)
        .head(n)
        .rename(columns={name_col: "Project", col: label})
    )
    if plot_df.empty:
        return _empty_chart("Run data collection first")
    return (
        alt.Chart(plot_df)
        .mark_bar(color=color, cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
        .encode(
            x=alt.X(f"{label}:Q", title=label),
            y=alt.Y("Project:N", sort="-x", title=""),
            tooltip=[
                alt.Tooltip("Project:N", title="Project"),
                alt.Tooltip(f"{label}:Q", title=label, format=","),
            ],
        )
        .properties(height=max(200, n * 18), width="container")
    )


def _altair_health_heatmap(data: pd.DataFrame) -> alt.Chart:
    if data.empty:
        return _empty_chart()
    health_cols = [c for c in FEATURES_HEALTH if c in data.columns]
    if not health_cols:
        return _empty_chart("Run data collection first")

    name_col = "name" if "name" in data.columns else data.columns[0]
    rows = []
    for _, row in data.iterrows():
        score = 0
        for col in health_cols:
            v = row.get(col)
            present = 1 if (v is True or str(v).lower() in ("true", "1")) else 0
            score += present
            rows.append({
                "Project":      str(row[name_col]),
                "Health Check": FEATURE_LABELS.get(col, col),
                "Present":      present,
                "_score":       0,  # filled below
            })

    df_h = pd.DataFrame(rows)
    # Compute per-project total score for y-axis sort
    scores = (
        df_h.groupby("Project")["Present"].sum()
        .reset_index()
        .rename(columns={"Present": "_score"})
    )
    df_h = df_h.drop(columns=["_score"]).merge(scores, on="Project")
    df_h["Present_label"] = df_h["Present"].map({1: "✓", 0: ""})

    base = alt.Chart(df_h).encode(
        x=alt.X("Health Check:N",
                 sort=list(FEATURE_LABELS.values()),
                 axis=alt.Axis(labelAngle=-30, labelLimit=120)),
        y=alt.Y("Project:N",
                 sort=alt.EncodingSortField(field="_score", order="descending")),
    )

    heatmap = base.mark_rect().encode(
        color=alt.Color(
            "Present:Q",
            scale=alt.Scale(domain=[0, 1], range=["#f0f0f0", NF_GREEN]),
            legend=None,
        ),
        tooltip=[
            alt.Tooltip("Project:N"),
            alt.Tooltip("Health Check:N"),
            alt.Tooltip("Present:Q", title="Present (1=yes, 0=no)"),
        ],
    )
    text = base.mark_text(fontSize=11, color="#333").encode(
        text="Present_label:N"
    )

    n_projects = data[name_col].nunique()
    return (
        (heatmap + text)
        .properties(height=max(300, n_projects * 18), width="container")
    )


def _altair_health_vs_stars(data: pd.DataFrame) -> alt.Chart:
    if (data.empty
            or "health_percentage" not in data.columns
            or "stargazers_count" not in data.columns):
        return _empty_chart("Run data collection first")

    plot_df = data.copy()
    plot_df["Health %"] = pd.to_numeric(plot_df["health_percentage"], errors="coerce")
    plot_df["Stars"]    = pd.to_numeric(plot_df["stargazers_count"],  errors="coerce")
    plot_df = plot_df.dropna(subset=["Health %", "Stars"])
    if plot_df.empty:
        return _empty_chart("Run data collection first")

    plot_df["Language"] = (
        plot_df["language_tags"]
        .fillna("Other")
        .str.split("|")
        .str[0]
        .str.strip()
    )
    name_col = "name" if "name" in plot_df.columns else plot_df.columns[0]

    return (
        alt.Chart(plot_df)
        .mark_circle(size=90, opacity=0.85, stroke="white", strokeWidth=0.8)
        .encode(
            x=alt.X("Stars:Q",    title="GitHub Stars",
                    scale=alt.Scale(zero=False)),
            y=alt.Y("Health %:Q", title="GitHub Health %",
                    scale=alt.Scale(domain=[0, 105])),
            color=alt.Color("Language:N",
                            legend=alt.Legend(title="Language")),
            tooltip=[
                alt.Tooltip(f"{name_col}:N", title="Project"),
                alt.Tooltip("Stars:Q",        format=","),
                alt.Tooltip("Health %:Q",     format=".0f"),
                alt.Tooltip("Language:N"),
            ],
        )
        .properties(height=320, width="container")
        .interactive()
    )


@reactive.calc
def _security_table_df() -> pd.DataFrame:
    """
    Join filtered project names onto scorecard data, return display-ready DataFrame.
    Scores of -1 (check not applicable) are shown as N/A.
    Sorted by Total Score descending.
    """
    if df_security.empty:
        return pd.DataFrame()

    filtered_names = set(filtered_df()["name"].tolist()) if "name" in filtered_df().columns else set()

    # Build display columns
    display_cols = {}
    for label, col in SCORECARD_CHECKS:
        if col in df_security.columns:
            display_cols[col] = label

    keep = [c for c in ["name", "github_repo"] + list(display_cols.keys())
            if c in df_security.columns]
    out = df_security[keep].copy()

    # Filter to projects currently shown in the sidebar filters
    if filtered_names and "name" in out.columns:
        out = out[out["name"].isin(filtered_names)]

    out = out.rename(columns={**{"name": "Project", "github_repo": "GitHub repo"},
                               **display_cols})

    # Convert scores: -1 → pd.NA, then format
    for label, _ in SCORECARD_CHECKS:
        if label in out.columns:
            out[label] = pd.to_numeric(out[label], errors="coerce")
            out[label] = out[label].where(out[label] != -1, other=pd.NA)

    # Sort by total score
    if "Total Score" in out.columns:
        out = out.sort_values(
            "Total Score",
            key=lambda s: pd.to_numeric(s, errors="coerce"),
            ascending=False,
            na_position="last",
        )

    # Round for display
    for label, _ in SCORECARD_CHECKS:
        if label in out.columns:
            out[label] = out[label].map(
                lambda v: f"{float(v):.1f}" if pd.notna(v) else "N/A"
            )

    return out


def _altair_scorecard_averages(data: pd.DataFrame) -> alt.Chart:
    """Horizontal bar chart of average scorecard score per check (excluding -1)."""
    if data.empty:
        return _empty_chart("No scorecard data")

    rows = []
    for label, _ in SCORECARD_CHECKS:
        if label == "Total Score" or label not in data.columns:
            continue
        s = pd.to_numeric(data[label].replace("N/A", pd.NA), errors="coerce").dropna()
        if len(s) > 0:
            rows.append({"Check": label, "Average": round(float(s.mean()), 2),
                         "n": len(s)})

    if not rows:
        return _empty_chart("No numeric scores yet")

    df_avg = pd.DataFrame(rows).sort_values("Average", ascending=True)

    # Colour: green ≥7, orange 4–7, red <4
    def _color(v):
        if v >= 7: return NF_GREEN
        if v >= 4: return NF_ORANGE
        return NF_RED

    df_avg["color"] = df_avg["Average"].map(_color)
    df_avg["color_label"] = df_avg["Average"].map(
        lambda v: "≥7 (good)" if v >= 7 else ("4–6 (fair)" if v >= 4 else "<4 (risk)")
    )

    color_scale = alt.Scale(
        domain=["≥7 (good)", "4–6 (fair)", "<4 (risk)"],
        range=[NF_GREEN, NF_ORANGE, NF_RED],
    )

    return (
        alt.Chart(df_avg)
        .mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
        .encode(
            x=alt.X("Average:Q", title="Average score (0–10)",
                    scale=alt.Scale(domain=[0, 10])),
            y=alt.Y("Check:N", sort="-x", title=""),
            color=alt.Color("color_label:N",
                            scale=color_scale,
                            legend=alt.Legend(title="Rating")),
            tooltip=[
                alt.Tooltip("Check:N"),
                alt.Tooltip("Average:Q", format=".2f"),
                alt.Tooltip("n:Q", title="Projects scored"),
            ],
        )
        .properties(height=420, width="container")
    )


def _lang_spark_ui(sel):
    top = _safe(sel.get("top_languages"), "")
    if not top or top == "—":
        return ui.p("Language breakdown not yet collected.", class_="text-muted small")
    langs = [l.strip() for l in top.split("|") if l.strip()]
    gh_repo = _safe(sel.get("github_repo"), "")
    link = f"https://github.com/{gh_repo}#readme" if gh_repo and gh_repo != "—" else ""
    return ui.div(
        ui.p("Top languages used in this repository:"),
        ui.div(*[ui.span(l, class_="tag-chip") for l in langs]),
        ui.br(),
        ui.p(ui.tags.a("View full breakdown on GitHub →",
                       href=link, target="_blank")) if link else ui.span(""),
    )
