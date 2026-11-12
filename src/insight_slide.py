"""One-slide business insight, generated from the real aspect results.

Nothing here is hard-coded: every number and quote comes from the analysis the
user just ran. The slide is a single self-contained HTML file (16:9, scales to
any window, print-to-PDF friendly) that can be screenshotted or dropped into a
presentation.
"""

from __future__ import annotations

import html
from datetime import date

import pandas as pd

MIN_MENTIONS = 3  # aspects mentioned fewer times are too thin to rank

ACTIONS: dict[str, str] = {
    "delivery": "Audit courier performance and packaging, and promise delivery times customers can trust.",
    "price": "Review pricing and bundles, and make the value of the product clearer to buyers.",
    "product quality": "Investigate defect and return reasons for the affected products and tighten quality checks.",
    "battery": "Check battery-life claims and charger quality, and state real-world battery life honestly.",
    "screen": "Review display quality and brightness complaints with the product team.",
    "sound": "Review speaker and audio complaints with the product team.",
    "customer support": "Cut first-response time and train agents on the most common complaints.",
    "staff": "Coach front-line staff and share examples of the interactions customers praise.",
    "app / website": "Prioritise bug fixes in the app, login and checkout flow.",
    "refund / return": "Simplify refunds and returns and publish a clear processing time.",
    "process": "Remove waiting steps and publish expected timelines for each stage.",
    "ease of use": "Improve onboarding and setup instructions.",
}
DEFAULT_ACTION = "Read the most negative comments for this aspect, then agree an owner and a fix date."


def _quote(aspect_df: pd.DataFrame, aspect: str, sentiment: str) -> str:
    """The most confident, readable clause for an aspect and sentiment."""
    rows = aspect_df[(aspect_df["aspect"] == aspect) & (aspect_df["aspect_sentiment"] == sentiment)]
    rows = rows[rows["clause"].astype(str).str.split().str.len().between(4, 28)]
    if rows.empty:
        return ""
    return str(rows.sort_values("confidence", ascending=False).iloc[0]["clause"]).strip()


def build_insight(summary: pd.DataFrame, aspect_df: pd.DataFrame, total_reviews: int) -> dict | None:
    """Turn the per-aspect summary into the facts the slide shows."""
    if summary is None or summary.empty:
        return None
    rated = summary[summary["mentions"] >= MIN_MENTIONS]
    if rated.empty:
        rated = summary
    worst = rated.sort_values(["negative_share", "mentions"], ascending=False).iloc[0]
    best = rated.sort_values(["positive_share", "mentions"], ascending=False).iloc[0]
    has_problem = float(worst["negative_share"]) > 0

    if not has_problem:
        headline = "No aspect stands out as a problem in this feedback"
    elif str(worst["aspect"]) == str(best["aspect"]):
        headline = f"{str(worst['aspect']).title()} is the aspect customers talk about most, and opinion is split"
    else:
        headline = f"Customers praise {best['aspect']}, but {worst['aspect']} drives the most complaints"

    top_negative = (
        rated[rated["negative_share"] > 0]
        .sort_values(["negative_share", "mentions"], ascending=False)
        .head(5)[["aspect", "negative_share", "mentions"]]
        .to_dict("records")
    )
    return {
        "headline": headline,
        "has_problem": has_problem,
        "worst": {
            "aspect": str(worst["aspect"]),
            "share": float(worst["negative_share"]),
            "mentions": int(worst["mentions"]),
            "quote": _quote(aspect_df, str(worst["aspect"]), "negative"),
        },
        "best": {
            "aspect": str(best["aspect"]),
            "share": float(best["positive_share"]),
            "mentions": int(best["mentions"]),
            "quote": _quote(aspect_df, str(best["aspect"]), "positive"),
        },
        "top_negative": top_negative,
        "action": ACTIONS.get(str(worst["aspect"]), DEFAULT_ACTION) if has_problem else "Keep monitoring; no urgent fix is indicated by this data.",
        "total_reviews": int(total_reviews),
        "total_mentions": int(len(aspect_df)),
        "reviews_with_aspect": int(aspect_df["feedback_id"].nunique()),
    }


def render_slide_html(insight: dict, title: str = "Customer Feedback Insight") -> str:
    """A standalone 16:9 HTML slide for the insight dictionary."""
    e = html.escape
    worst, best = insight["worst"], insight["best"]

    bars = ""
    for row in insight["top_negative"]:
        pct = row["negative_share"] * 100
        bars += (
            f'<div class="bar-row"><div class="bar-name">{e(str(row["aspect"]).title())}</div>'
            f'<div class="bar-track"><div class="bar-fill" style="width:{pct:.0f}%"></div></div>'
            f'<div class="bar-val">{pct:.0f}%<span> of {int(row["mentions"])}</span></div></div>'
        )
    if not bars:
        bars = '<div class="muted">No aspect has negative mentions.</div>'

    def quote_block(text: str) -> str:
        return f'<div class="quote">&ldquo;{e(text)}&rdquo;</div>' if text else ""

    share_reviews = insight["reviews_with_aspect"] / insight["total_reviews"] if insight["total_reviews"] else 0
    pain_label = "Biggest pain point" if insight["has_problem"] else "Most negative aspect"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{e(title)}</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  html,body{{background:#e2e8f0;font-family:"Segoe UI",Roboto,Helvetica,Arial,sans-serif;color:#0f172a}}
  body{{display:flex;align-items:center;justify-content:center;min-height:100vh}}
  .slide{{container-type:inline-size;width:min(100vw,177.78vh);aspect-ratio:16/9;background:#fff;display:flex;flex-direction:column;overflow:hidden}}
  .head{{background:#0b1324;color:#fff;padding:2.2cqw 3.2cqw 2cqw}}
  .eyebrow{{font-size:1cqw;letter-spacing:.16em;text-transform:uppercase;color:#93c5fd;font-weight:700}}
  h1{{font-size:2.9cqw;line-height:1.15;margin-top:.7cqw;font-weight:700;letter-spacing:-.01em}}
  .main{{flex:1;display:grid;grid-template-columns:1.05fr 1fr;gap:2.2cqw;padding:2.2cqw 3.2cqw 1.2cqw}}
  .card{{border-radius:1.1cqw;padding:1.6cqw 1.8cqw;border:1px solid #e2e8f0}}
  .card.bad{{background:#fef2f2;border-color:#fecaca}}
  .card.good{{background:#ecfdf5;border-color:#bbf7d0;margin-top:1.4cqw}}
  .label{{font-size:.95cqw;letter-spacing:.12em;text-transform:uppercase;font-weight:700;color:#64748b}}
  .big{{font-size:4.6cqw;font-weight:800;line-height:1;margin:.5cqw 0 .2cqw}}
  .bad .big{{color:#b91c1c}} .good .big{{color:#0f766e}}
  .what{{font-size:1.5cqw;font-weight:700}}
  .sub{{font-size:1.15cqw;color:#475569;margin-top:.2cqw}}
  .quote{{font-size:1.2cqw;font-style:italic;color:#334155;margin-top:.9cqw;border-left:.3cqw solid #cbd5e1;padding-left:.9cqw}}
  .panel-title{{font-size:1.2cqw;font-weight:700;margin-bottom:1cqw}}
  .bar-row{{display:grid;grid-template-columns:11cqw 1fr 8cqw;align-items:center;gap:.8cqw;margin-bottom:.9cqw}}
  .bar-name{{font-size:1.2cqw;font-weight:600}}
  .bar-track{{background:#f1f5f9;border-radius:.6cqw;height:1.5cqw;overflow:hidden}}
  .bar-fill{{background:#dc2626;height:100%;border-radius:.6cqw}}
  .bar-val{{font-size:1.2cqw;font-weight:700;text-align:right}} .bar-val span{{font-weight:400;color:#64748b;font-size:.95cqw}}
  .action{{margin-top:1.4cqw;background:#0b1324;color:#fff;border-radius:1.1cqw;padding:1.4cqw 1.7cqw}}
  .action .label{{color:#93c5fd}} .action p{{font-size:1.35cqw;line-height:1.35;margin-top:.5cqw}}
  .muted{{color:#64748b;font-size:1.2cqw}}
  .foot{{display:flex;justify-content:space-between;gap:2cqw;padding:.9cqw 3.2cqw 1.3cqw;font-size:.95cqw;color:#64748b;border-top:1px solid #e2e8f0}}
  @media print{{@page{{size:13.333in 7.5in;margin:0}} html,body{{background:#fff}} body{{display:block;min-height:0}} .slide{{width:100vw;height:100vh}}}}
</style></head><body>
<div class="slide">
  <div class="head"><div class="eyebrow">{e(title)}</div><h1>{e(insight["headline"])}</h1></div>
  <div class="main">
    <div>
      <div class="card bad">
        <div class="label">{pain_label}</div>
        <div class="big">{worst["share"]:.0%}</div>
        <div class="what">of {e(worst["aspect"])} mentions are negative</div>
        <div class="sub">{worst["mentions"]} mentions in total</div>
        {quote_block(worst["quote"])}
      </div>
      <div class="card good">
        <div class="label">Strongest asset</div>
        <div class="big">{best["share"]:.0%}</div>
        <div class="what">of {e(best["aspect"])} mentions are positive</div>
        <div class="sub">{best["mentions"]} mentions in total</div>
        {quote_block(best["quote"])}
      </div>
    </div>
    <div>
      <div class="panel-title">Where the complaints concentrate (share of each aspect&rsquo;s mentions that are negative)</div>
      {bars}
      <div class="action"><div class="label">Recommended next step</div><p>{e(insight["action"])}</p></div>
    </div>
  </div>
  <div class="foot">
    <span>Based on {insight["total_reviews"]:,} reviews &middot; {insight["total_mentions"]:,} aspect mentions &middot; {share_reviews:.0%} of reviews mention at least one aspect</span>
    <span>Aspect sentiment: DistilBERT scored per clause &middot; indicative, validate before acting &middot; {date.today():%d %b %Y}</span>
  </div>
</div>
</body></html>
"""
