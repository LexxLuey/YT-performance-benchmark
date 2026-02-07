# Script Workflow Explanation

## High-Level Flow (Simplified)

The script is a **single-command analysis tool**:

# Step 1: Create videos.txt with 4 URLs

```txt
"https://youtube.com/watch?v=VIDEO1"
"https://youtube.com/watch?v=VIDEO2"
"https://youtube.com/watch?v=VIDEO3"
"https://youtube.com/watch?v=VIDEO4"

```

# Step 2: Run the script once

```bash
python app.py

# Output: Analyzed videos, extracted signals, generated insights
```

That's it. The script handles everything:

1. Fetches metadata from YouTube
2. Auto-classifies videos as high/low performance
3. Downloads videos
4. Extracts signals
5. Compares cohorts
6. Generates visualizations & insights

---

## Step-by-Step Execution Flow

### **Step 1: Read Input**

```
User creates: videos.txt (4 URLs, one per line)

Script reads: videos.txt
├─ Video 1 URL
├─ Video 2 URL
├─ Video 3 URL
└─ Video 4 URL

Output: "Found 4 video URLs"
```

---

### **Step 2: Fetch Metadata**

```
For each URL:
├─ Use yt-dlp to fetch metadata (no download yet)
├─ Extract: title, duration, views, likes, comments, subscriber_count
└─ Print: "Video 1: 50,000 views, 2,500 likes, 125 comments"

Output: Prints all 4 videos with metadata
```

---

### **Step 3: Auto-Classify Performance**

```
For each video, calculate engagement score:
├─ Like ratio = (likes / views) × 100
├─ Comment density = comments / (views / 1000)
├─ Viral ratio = views / subscriber_count
├─ Composite = 0.4 × like_ratio + 0.4 × comment_density + 0.2 × (viral_ratio / 10)
└─ Classification: score > 3.5 = HIGH, score < 1.5 = LOW

Output: "Video 1: 4.2 (HIGH performer)" etc.
```

---

### **Step 4: Generate videos.json**

```
Script creates: videos.json with:
├─ Metadata (title, views, likes, comments, duration)
├─ Engagement metrics (like_ratio, comment_density, viral_ratio)
├─ Justification (auto-generated explanation for classification)
├─ Performance classification (high or low)
└─ URLs for download

Output: "Generated videos.json"
```

---

### **Step 5: Download Videos**

```
For each video URL:
├─ Use yt-dlp to download MP4 file
├─ Store in ./videos/ directory
├─ Print progress: "Downloaded video_1.mp4 (12:34 duration)"
└─ Handle errors gracefully (network timeout, unavailable, etc.)
```

---

### **Step 6: Extract Signals (Per Video)**

```
For each downloaded video:
├─ Extract Frame-Based Signals (OpenCV):
│  ├─ extract_scene_cuts() → cuts per minute
│  ├─ extract_motion_intensity() → avg frame variance
│  └─ extract_color_variance() → HSV variance
│
└─ Extract Audio Signals (librosa):
   ├─ extract_silence_ratio() → % dead air
   └─ extract_loudness_variance() → RMS std dev

Output per video:
{
  "video_id": "video_1",
  "scene_cuts_per_min": 8.2,
  "motion_intensity": 0.45,
  "color_variance": 125.3,
  "silence_ratio": 0.08,
  "loudness_variance": 0.32
}
```

---

### **Step 7: Aggregate Results**

```
Combine all signals into pandas DataFrame:

| video_id | performance | scene_cuts | motion | color_var | silence | loudness |
|----------|-------------|-----------|--------|-----------|---------|----------|
| video_1  | high       | 8.2       | 0.45   | 125.3     | 0.08    | 0.32     |
| video_2  | high       | 7.9       | 0.42   | 118.5     | 0.07    | 0.29     |
| video_3  | low        | 3.1       | 0.18   | 62.1      | 0.22    | 0.14     |
| video_4  | low        | 2.8       | 0.16   | 58.3      | 0.25    | 0.12     |

Output: Prints summary statistics (mean ± std for each group)
```

---

### **Step 8: Statistical Comparison**

```
For each signal:
├─ Compute mean & std for high-performing cohort
├─ Compute mean & std for low-performing cohort
├─ Run t-test to check significance (p < 0.05)
└─ Print results:
   "Scene Cuts: High = 8.1 ± 0.2 cuts/min, Low = 3.0 ± 0.2 cuts/min (p=0.001) **"
```

---

### **Step 9: Visualize Comparisons**

```
Create matplotlib figure with 5 subplots (one per signal):
├─ Subplot 1: Scene Cuts (box plot: high vs. low)
├─ Subplot 2: Motion Intensity (box plot: high vs. low)
├─ Subplot 3: Color Variance (box plot: high vs. low)
├─ Subplot 4: Silence Ratio (box plot: high vs. low)
└─ Subplot 5: Loudness Variance (box plot: high vs. low)

Save as: ./output/signals_comparison.png
Output: "Saved visualization to output/signals_comparison.png"
```

---

### **Step 10: Generate Insights**

```
For each signal with statistically significant difference:
├─ Extract quantified gap (e.g., 8.1 vs. 3.0)
├─ Write actionable insight (plain English)
└─ Suggest testable recommendation

Save as: INSIGHTS.md

Example:
"Scene Cut Frequency
- High performers: 8.1 ± 0.2 cuts/min
- Low performers: 3.0 ± 0.2 cuts/min
- Insight: High-performing videos introduce a visual change every 7–9 seconds, 
  while low-performing videos remain static for 19–20 seconds. 
  Frequent cuts maintain viewer attention.
- Recommendation: Aim for 1 cut every 8–10 seconds."
```

---

### **Step 11: Final Output Summary**

```
Print to console & save to analysis_report.txt:

═══════════════════════════════════════════
ANALYSIS COMPLETE
═══════════════════════════════════════════
Videos Analyzed: 4 (2 high-performing, 2 low-performing)
Niche: [auto-detected from metadata]
Signals Extracted: 5
Statistically Significant Differences: 4/5
Top Insight: "Scene cut frequency is 2.7x higher in high-performers"

Files Generated:
✓ videos.json (metadata + classifications)
✓ ./videos/ (downloaded MP4s)
✓ output/signals_comparison.png (visualizations)
✓ output/analysis_report.txt (statistics)
✓ output/results.csv (detailed metrics per video)
✓ INSIGHTS.md (actionable findings)
```

---

## Data Flow Diagram

```
videos.txt (4 URLs)
    ↓
[Fetch Metadata] ←─ yt-dlp
    ↓
[Auto-Classify] ←─ engagement formula
    ↓
videos.json (generated automatically)
    ↓
[Download Videos] ←─ yt-dlp
    ↓
./videos/ (MP4 files)
    ↓
[Extract Signals] ←─ OpenCV (visual) + librosa (audio)
    ↓
DataFrame (5 signals × 4 videos)
    ↓
[Compare Cohorts] ←─ scipy.stats (t-tests)
    ↓
[Visualize] ←─ matplotlib
    ↓
./output/signals_comparison.png
    ↓
[Generate Insights] ←─ manual interpretation of metrics
    ↓
INSIGHTS.md
```

---

## Code Structure Preview

```python
# app.py

# Phase 1: Input & Auto-Classification
def load_video_urls() -> List[str]:
    """Load 4 URLs from videos.txt"""
    ...

def fetch_metadata(url: str) -> Dict:
    """Fetch metadata from YouTube via yt-dlp"""
    ...

def calculate_engagement_score(views, likes, comments, subs) -> float:
    """Calculate composite engagement score for auto-classification"""
    ...

def auto_classify_videos(videos: List[Dict]) -> List[Dict]:
    """Classify videos as high/low based on engagement score"""
    ...

def generate_videos_json(videos: List[Dict]) -> None:
    """Auto-generate videos.json with metadata & classifications"""
    ...

# Phase 2: Signal Extraction
def extract_scene_cuts(video_path: str) -> float:
    ...
def extract_motion_intensity(video_path: str) -> float:
    ...
def extract_color_variance(video_path: str) -> float:
    ...
def extract_silence_ratio(video_path: str) -> float:
    ...
def extract_loudness_variance(video_path: str) -> float:
    ...

# Phase 3-5: Analysis & Insights
def process_all_videos(videos: List[Dict]) -> pd.DataFrame:
    ...

def compare_cohorts(df: pd.DataFrame) -> None:
    ...

def generate_insights(df: pd.DataFrame) -> None:
    ...

# Main entry point
if __name__ == "__main__":
    urls = load_video_urls()
    videos = [fetch_metadata(url) for url in urls]
    videos = auto_classify_videos(videos)
    generate_videos_json(videos)
    results = process_all_videos(videos)
    compare_cohorts(results)
    generate_insights(results)
    print("✓ Analysis complete!")
```
