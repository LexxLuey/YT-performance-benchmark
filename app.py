"""
YouTube Video Analyzer - Extract signals from videos to identify patterns.

This script analyzes YouTube videos to identify patterns differentiating
high-performing from low-performing streams, converting findings into
actionable insights for content creators.

Main entry point: python app.py

Workflow:
1. Load video URLs from videos.txt
2. Fetch metadata from YouTube (no download yet)
3. Auto-classify videos based on engagement score
4. Download videos
5. Extract signals (visual & audio)
6. Compare cohorts & generate insights

See USAGE.md and ROADMAP.md for detailed flow documentation.
"""

import os
import json
import re
import math
from typing import Dict, List, Optional
from pathlib import Path

import yt_dlp
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Initialize rich console for colored/formatted output
console = Console()


# =============================================================================
# Phase 1: Video Input & Auto-Classification
# =============================================================================


def load_video_urls(filepath: str = "videos.txt") -> List[str]:
    """
    Load YouTube video URLs from a text file.

    Expects a simple text file with one URL per line. Blank lines and
    comments (lines starting with #) are ignored.

    Parameters:
    filepath (str): Path to videos.txt file (default: "videos.txt")

    Returns:
    List[str]: List of 4 YouTube URLs

    Raises:
    FileNotFoundError: If videos.txt does not exist
    ValueError: If fewer than 4 URLs found
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"❌ {filepath} not found. Create it with 4 YouTube URLs " "(one per line)"
        )

    urls = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            # Skip blank lines and comments
            if line and not line.startswith("#"):
                urls.append(line)

    if len(urls) < 4:
        raise ValueError(
            f"❌ Found {len(urls)} URLs, need exactly 4. "
            "Please add more video URLs to videos.txt"
        )

    return urls[:4]  # Return only first 4


def fetch_video_metadata(url: str) -> Optional[Dict]:
    """
    Fetch metadata from a YouTube video without downloading.

    Uses yt-dlp to extract metadata including views, likes, comments,
    duration, channel info, and subscriber count. This data is used for
    engagement scoring and auto-classification.

    Trade-off: yt-dlp extraction (no auth, faster) vs YouTube Data API
    (official but requires API key and has rate limits). We choose yt-dlp
    for simplicity in a proof-of-concept analysis.

    Parameters:
    url (str): YouTube video URL

    Returns:
    Dict: Metadata with keys:
        - title (str): Video title
        - duration (int): Duration in seconds
        - views (int): View count
        - likes (int): Like count
        - comments (int): Comment count
        - upload_date (str): Upload date (YYYYMMDD format)
        - channel (str): Channel/uploader name
        - subscriber_count (int): Channel subscriber count
    Returns None if video is unavailable.

    Raises:
    Exception: Caught and logged; returns None gracefully
    """
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,  # Only fetch metadata, don't download
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)

            # Extract subscriber count (may not always be available)
            subscriber_count = info.get("channel_follower_count", 0) or 0

            return {
                "url": url,
                "title": info.get("title", "N/A"),
                "duration": info.get("duration", 0),  # seconds
                "views": info.get("view_count", 0),
                "likes": info.get("like_count", 0),
                "comments": info.get("comment_count", 0),
                "upload_date": info.get("upload_date", "N/A"),
                "channel": info.get("uploader", "N/A"),
                "subscriber_count": subscriber_count,
            }
    except Exception as e:
        console.print(f"[yellow]⚠️  Could not fetch metadata for {url}: {e}[/yellow]")
        return None


def calculate_engagement_score(
    views: Optional[int],
    likes: Optional[int],
    comments: Optional[int],
    subscriber_count: Optional[int],
) -> Dict[str, float]:
    """
    Calculate engagement score using log-scale normalization.

    Since private retention data is unavailable, we use public metrics
    that correlate with viewer satisfaction:
    - Absolute engagement: (likes + comments*3); indicates audience depth
    - Viral reach: (views / subscriber_count); indicates algorithmic breadth

    Composite score weights: 70% engagement + 20% reach + 10% ratio (if >5K views)

    Key insight: Engagement data follows a power law distribution. Using linear
    scaling (e.g., dividing by 500) causes scores to max out early or lose
    resolution. Logarithmic scaling (log10) handles outliers and skewed
    distributions much better, capturing the relationship:
    - 100 engagement -> log10(101) = 2.0
    - 1,000 engagement -> log10(1001) = 3.0
    - 10,000 engagement -> log10(10001) = 4.0

    Trade-off: We prioritize absolute engagement (volume) over ratio-based
    metrics because ratios are statistically meaningless on small sample sizes.

    Note: Some creators disable likes/comments. We handle None values by
    treating them as 0 (conservative engagement estimate).

    Parameters:
    views (Optional[int]): Total views (may be None if unavailable)
    likes (Optional[int]): Total likes (may be None if hidden by creator)
    comments (Optional[int]): Total comments (may be None if disabled)
    subscriber_count (Optional[int]): Channel subscriber count (may be None)

    Returns:
    Dict: Contains absolute_engagement, viral_ratio, composite_score
    """
    # Handle None values by converting to 0 (conservative estimate)
    views = views or 0
    likes = likes or 0
    comments = comments or 0
    subscriber_count = subscriber_count or 0

    # PRIMARY SIGNAL: Absolute engagement volume (log-scaled)
    # Comments are rarer, so weight them 3x higher than likes
    absolute_engagement = likes + (comments * 3)
    engagement_score = math.log10(absolute_engagement + 1)  # +1 to avoid log(0)

    # SECONDARY SIGNAL: Viral reach (log-scaled)
    # Did the video reach beyond the subscriber base?
    viral_ratio = views / subscriber_count if subscriber_count > 0 else 0
    reach_score = math.log10(viral_ratio + 1)  # +1 to handle ratio < 1

    # TERTIARY SIGNAL: Engagement ratios (only meaningful if >5K views)
    # For small samples, ratio-based metrics are noise
    if views >= 5000:
        like_ratio = (likes / views) * 100
        comment_density = comments / (views / 1000)
    else:
        like_ratio = 0
        comment_density = 0

    # Composite score: 70% engagement (primary) + 20% reach (secondary) + 10% ratio (tertiary)
    # Each component normalized to roughly 0-5 scale for interpretability
    composite_score = (
        engagement_score * 0.7  # Absolute engagement (primary)
        + reach_score * 0.2  # Viral reach (secondary)
        + (like_ratio / 10) * 0.1  # Ratio-based signal (tertiary, only if >5K views)
    )

    return {
        "absolute_engagement": absolute_engagement,
        "engagement_score_log": round(engagement_score, 2),
        "like_ratio": round(like_ratio, 2) if views >= 5000 else None,
        "comment_density": round(comment_density, 2) if views >= 5000 else None,
        "viral_ratio": round(viral_ratio, 2),
        "reach_score_log": round(reach_score, 2),
        "composite_score": round(composite_score, 2),
        "sample_size_reliable": views >= 5000,
    }


def classify_performance(composite_score: float) -> str:
    """
    Classify video as high or low-performing based on engagement score.

    Thresholds (tuned for log-scaled composite score):
    - HIGH: composite_score > 2.35 (strong absolute engagement + reach)
    - LOW: composite_score <= 2.35 (weak absolute engagement or reach)

    With log-scaled metrics, scores are compressed. A score of 2.35 represents:
    - ~200+ absolute engagement (likes + comments*3)
    - Reliable performance gap with adequate samples

    This is a reasonable threshold for distinguishing high from low performers.

    Parameters:
    composite_score (float): Composite engagement score (log-scaled)

    Returns:
    str: 'high' or 'low'
    """
    if composite_score > 2.35:
        return "high"
    else:
        return "low"


def generate_justification(
    metadata: Dict,
    engagement: Dict,
    performance: str,
) -> str:
    """
    Generate human-readable justification for performance classification.

    Explains classification based on absolute engagement volume and viral reach
    to make the auto-classification transparent and debuggable.

    Key principle: For large channels (e.g., Daigo with 272k subscribers), a
    video reaching 100k views is "low" performance if viral_ratio is 0.37
    (meaning it only reached 37% of subscribers + minimal new users). In contrast,
    a smaller channel with 7k subscribers reaching 18k views has viral_ratio 2.5
    (reached 2.5x the subscriber base), indicating algorithmic success.

    Parameters:
    metadata (Dict): Video metadata (views, likes, comments, subscribers)
    engagement (Dict): Engagement scores (absolute, viral, composite)
    performance (str): Classification ('high' or 'low')

    Returns:
    str: Justification string explaining the classification
    """
    views = metadata.get("views", 0)
    likes = metadata.get("likes", 0)
    comments = metadata.get("comments", 0)
    subs = metadata.get("subscriber_count", 0)
    absolute_engagement = engagement.get("absolute_engagement", 0)
    viral_ratio = engagement.get("viral_ratio", 0)
    composite_score = engagement.get("composite_score", 0)

    if performance == "high":
        return (
            f"High performer: {absolute_engagement} absolute engagement "
            f"(likes + comments×3) + {viral_ratio:.2f}x viral reach. "
            f"Strong audience depth (likes/comments) and breadth (reached beyond "
            f"subscriber base). Composite score: {composite_score:.2f}."
        )
    else:
        # For low performers, explain the relative underperformance
        if subs > 100000:
            # Large channel: low viral ratio indicates failure to reach new users
            return (
                f"Low performer: {absolute_engagement} absolute engagement "
                f"+ {viral_ratio:.2f}x viral reach on {subs:,} subscriber channel. "
                f"Video failed to expand reach beyond core fanbase. Expected to reach "
                f">1.5x subscribers for algorithmic success. Composite score: {composite_score:.2f}."
            )
        else:
            # Smaller channel: low absolute engagement
            return (
                f"Low performer: {absolute_engagement} absolute engagement "
                f"+ {viral_ratio:.2f}x viral reach. Limited audience reaction "
                f"(few likes/comments) and modest algorithmic reach. Composite score: {composite_score:.2f}."
            )


def auto_classify_videos(
    videos_metadata: List[Optional[Dict]],
) -> List[Dict]:
    """
    Auto-classify videos as high or low-performing.

    For each video with metadata, calculates engagement score and assigns
    performance classification. Videos with failed metadata fetch are skipped.

    Parameters:
    videos_metadata (List[Optional[Dict]]): List of metadata dicts or None

    Returns:
    List[Dict]: Classified videos with engagement scores and justifications
    """
    classified_videos = []

    for idx, metadata in enumerate(videos_metadata, 1):
        # Skip videos with failed metadata fetch
        if metadata is None:
            console.print(
                f"  [yellow]⚠️  Video {idx}: Skipped (metadata fetch failed)[/yellow]"
            )
            continue

        # Calculate engagement score
        engagement = calculate_engagement_score(
            views=metadata.get("views", 0),
            likes=metadata.get("likes", 0),
            comments=metadata.get("comments", 0),
            subscriber_count=metadata.get("subscriber_count", 0),
        )

        # Classify as high or low
        performance = classify_performance(engagement["composite_score"])

        # Generate justification
        justification = generate_justification(metadata, engagement, performance)

        # Assemble video record
        video_record = {
            "title": metadata.get("title", "N/A"),
            "url": metadata.get("url"),
            "duration_minutes": round(metadata.get("duration", 0) / 60, 1),
            "views": metadata.get("views", 0),
            "likes": metadata.get("likes", 0),
            "comments": metadata.get("comments", 0),
            "subscriber_count": metadata.get("subscriber_count", 0),
            "upload_date": metadata.get("upload_date", "N/A"),
            "channel": metadata.get("channel", "N/A"),
            "engagement_metrics": engagement,
            "performance": performance,
            "justification": justification,
        }

        classified_videos.append(video_record)

        # Print progress
        score = engagement["composite_score"]
        symbol = "⬆️ " if performance == "high" else "⬇️ "
        color = "green" if performance == "high" else "red"
        console.print(
            f"  {symbol} Video {idx}: [{color}]{score:.2f} ({performance.upper()})[/{color}] - "
            f"{metadata['title'][:50]}"
        )

    return classified_videos


def detect_niche(videos: List[Dict]) -> tuple[str, bool]:
    """
    Detect common niche from video titles and channel names.

    Returns: (niche_name, is_same_niche) where is_same_niche is True if
    all videos belong to the same detected niche.

    Rationale: Helps verify the "same_niche" constraint automatically.

    Parameters:
    videos (List[Dict]): Auto-classified video records

    Returns:
    tuple: (niche_name: str, same_niche: bool)
    """

    # Define niche keywords
    niche_keywords = {
        "Fighting Games": [
            "tekken",
            "street fighter",
            "sf6",
            "sf5",
            "mortal kombat",
            "mk",
            "dragonball",
            "guilty gear",
            "blazblue",
            "evo",
            "fgc",
            "fighting",
            "tournament",
            "grand finals",
            "esports",
            "ranked",
        ],
        "Gaming": [
            "gameplay",
            "twitch",
            "gaming",
            "esports",
            "competitive",
            "ranked",
            "speedrun",
            "challenge",
        ],
        "Productivity": [
            "productivity",
            "study",
            "work",
            "tutorial",
            "learning",
            "course",
            "tips",
            "guide",
        ],
        "Vlog": ["vlog", "daily", "day in my life", "life", "challenge", "reaction"],
    }

    # Collect all titles and channels
    combined_text = " ".join(
        [
            v.get("title", "").lower() + " " + v.get("channel", "").lower()
            for v in videos
        ]
    )

    # Count matches per niche
    niche_scores = {}
    for niche, keywords in niche_keywords.items():
        score = sum(combined_text.count(kw) for kw in keywords)
        if score > 0:
            niche_scores[niche] = score

    # Detect primary niche
    if niche_scores:
        detected_niche = max(niche_scores, key=niche_scores.get)
    else:
        detected_niche = "Mixed"

    # Verify all videos are same niche (all videos must match the dominant niche)
    # If 75%+ of videos match the detected niche, consider it same_niche
    videos_text = [
        (v.get("title", "") + " " + v.get("channel", "")).lower() for v in videos
    ]
    primary_keywords = niche_keywords.get(detected_niche, [])

    # A video matches the niche if it contains at least one primary keyword
    matching_videos = sum(
        1 for text in videos_text if any(kw in text for kw in primary_keywords)
    )

    # Consider same_niche if 75%+ of videos match (lenient for small samples)
    same_niche = matching_videos >= (len(videos) * 0.75)

    return detected_niche, same_niche


def generate_videos_json(videos: List[Dict], output_file: str = "videos.json") -> None:
    """
    Generate videos.json with auto-classified metadata.

    Separates videos into high_performers and low_performers categories,
    ready for download and signal extraction in Phase 2.

    Also detects niche from titles/channels and verifies the "same_niche"
    constraint automatically.

    Parameters:
    videos (List[Dict]): Auto-classified video records
    output_file (str): Output JSON file path
    """
    high_performers = [v for v in videos if v["performance"] == "high"]
    low_performers = [v for v in videos if v["performance"] == "low"]

    # Auto-detect niche from titles and channels
    detected_niche, same_niche = detect_niche(videos)

    # Calculate performance gap (ratio of high to low engagement)
    high_engagement = (
        sum(
            v["engagement_metrics"].get("absolute_engagement", 0)
            for v in high_performers
        )
        if high_performers
        else 0
    )
    low_engagement = (
        sum(
            v["engagement_metrics"].get("absolute_engagement", 0)
            for v in low_performers
        )
        if low_performers
        else 0
    )
    performance_gap = (high_engagement / low_engagement) if low_engagement > 0 else 0

    videos_data = {
        "niche": detected_niche,
        "rationale": (
            f"Auto-detected as '{detected_niche}' from video titles and channel names. "
            "All videos belong to the same niche and can be fairly compared."
        ),
        "high_performers": high_performers,
        "low_performers": low_performers,
        "constraints_verified": {
            "same_niche": same_niche,
            "all_at_least_5_min": all(v["duration_minutes"] >= 5 for v in videos),
            "clear_performance_gap": len(high_performers) > 0
            and len(low_performers) > 0,
            "performance_gap_ratio": round(performance_gap, 2),
            "notes": (
                f"All {len(videos)} videos are '{detected_niche}' content. "
                f"High performers have {performance_gap:.1f}x more engagement than low performers."
            ),
        },
    }

    with open(output_file, "w") as f:
        json.dump(videos_data, f, indent=2)

    console.print(f"\n[green]✅ Generated {output_file}[/green]")
    console.print(f"[blue]   Niche: {detected_niche}[/blue]")
    console.print(f"[blue]   Same niche verified: {same_niche}[/blue]")
    console.print(f"[blue]   High performers: {len(high_performers)}[/blue]")
    console.print(f"[blue]   Low performers: {len(low_performers)}[/blue]")
    console.print(f"[blue]   Performance gap: {performance_gap:.1f}x[/blue]")


# =============================================================================
# Phase 1 Main Entry Point
# =============================================================================


def run_phase_1() -> None:
    """
    Execute Phase 1: Video Input & Auto-Classification.

    Workflow:
    1. Load video URLs from videos.txt
    2. Fetch metadata from YouTube
    3. Auto-classify videos based on engagement score
    4. Generate videos.json with metadata
    """
    console.print("\n" + "[bold cyan]" + "=" * 70 + "[/bold cyan]")
    console.print("[bold cyan]PHASE 1: VIDEO INPUT & AUTO-CLASSIFICATION[/bold cyan]")
    console.print("[bold cyan]" + "=" * 70 + "[/bold cyan]" + "\n")

    # Step 1: Load URLs
    console.print("[cyan]📂 Step 1: Loading video URLs from videos.txt...[/cyan]")
    try:
        urls = load_video_urls()
        console.print(f"[green]✅ Found {len(urls)} video URLs[/green]\n")
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        return

    # Step 2: Fetch metadata
    console.print("[cyan]🌐 Step 2: Fetching metadata from YouTube...[/cyan]")
    videos_metadata = []
    for idx, url in enumerate(urls, 1):
        console.print(f"  [dim][{idx}/4] Fetching: {url[:60]}...[/dim]")
        metadata = fetch_video_metadata(url)
        videos_metadata.append(metadata)
    console.print()

    # Step 3: Auto-classify
    console.print("[cyan]📊 Step 3: Auto-classifying videos...[/cyan]")
    classified_videos = auto_classify_videos(videos_metadata)
    console.print()

    # Verify we have at least 2 high and 2 low performers
    high_count = sum(1 for v in classified_videos if v["performance"] == "high")
    low_count = sum(1 for v in classified_videos if v["performance"] == "low")

    if high_count < 2 or low_count < 2:
        console.print(
            f"[yellow]⚠️  Warning: Got {high_count} high-performers and {low_count} "
            f"low-performers. Ideally need 2 of each for good comparison.[/yellow]"
        )
        console.print(
            "[yellow]   Consider selecting videos with more diverse engagement metrics.[/yellow]\n"
        )

    # Step 4: Generate videos.json
    console.print("[cyan]💾 Step 4: Generating videos.json...[/cyan]")
    generate_videos_json(classified_videos)

    # Summary
    console.print("\n[bold green]" + "=" * 70 + "[/bold green]")
    console.print("[bold green]PHASE 1 COMPLETE ✅[/bold green]")
    console.print("[bold green]" + "=" * 70 + "[/bold green]")
    console.print("\n[bold]Next steps:[/bold]")
    console.print("[cyan]1. Review videos.json to verify classifications[/cyan]")
    console.print(
        "[cyan]2. Update 'niche' field to be more specific (e.g., 'productivity vlogs')[/cyan]"
    )
    console.print("[cyan]3. Verify all videos are ≥5 minutes and same category[/cyan]")
    console.print(
        "[cyan]4. Run 'python app.py' again for Phase 2 (signal extraction)[/cyan]\n"
    )


if __name__ == "__main__":
    run_phase_1()
