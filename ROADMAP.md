# Implementation Roadmap

## Overview

Comparative analysis of 4 YouTube videos (2 high-performing, 2 low-performing) to extract interpretable signals and generate actionable insights for content creators. **Target: 2 days, no over-engineering.**

---

## Phase 1: Video Input & Auto-Classification

**Goal**: Provide 4 video URLs; script auto-fetches metadata and classifies performance.

### Ultra-Simple User Flow

1. Create `videos.txt` with 4 YouTube URLs (one per line):

   ```
   https://youtube.com/watch?v=VIDEO1
   https://youtube.com/watch?v=VIDEO2
   https://youtube.com/watch?v=VIDEO3
   https://youtube.com/watch?v=VIDEO4
   ```

2. Run `python app.py`

3. Script automatically:
   - Fetches metadata (views, likes, comments) via yt-dlp
   - Calculates engagement scores (like ratio, comment density, viral ratio)
   - Auto-classifies videos as high or low-performing
   - Generates `videos.json` with all metadata
   - Proceeds to signal extraction & analysis

**Auto-Classification Formula** (see [copilot-instructions.md](.github/copilot-instructions.md) for details):

- **Like ratio**: (likes / views) × 100; >4% = high engagement
- **Comment density**: (comments / (views / 1000)); >1 = high engagement
- **Viral ratio**: views / subscriber_count; >2 = algorithmic reach
- **Composite score**: 0.4 × like_ratio + 0.4 × comment_density + 0.2 × (viral_ratio / 10)
- **High performer**: composite_score > 3.5
- **Low performer**: composite_score < 1.5

**Deliverable**: Auto-generated `videos.json` with justified classifications

---

## Phase 2: Signal Extraction

**Goal**: Extract 3–5 interpretable signals from video/audio with documented rationale.

### Visual Signals (OpenCV)

1. **Scene Cut Frequency**
   - **What**: Number of cuts/transitions per minute
   - **How**: Frame histogram comparison (detect large differences between consecutive frames)
   - **Why**: High-performing content cuts frequently to maintain viewer attention
   - **Code location**: `extract_scene_cuts(video_path: str) -> float`

2. **Motion Intensity**
   - **What**: Average frame variance per second (optical flow magnitude)
   - **How**: Compute frame differences or histogram variance across frames
   - **Why**: Dynamic content (camera movement, on-screen action) signals engagement
   - **Code location**: `extract_motion_intensity(video_path: str) -> float`

3. **Color/Brightness Variance**
   - **What**: Variance in HSV color space over time
   - **How**: Sample frames at regular intervals (10 fps), compute HSV histogram variance
   - **Why**: Visual stimulation and editing style indicator
   - **Code location**: `extract_color_variance(video_path: str) -> float`

### Audio Signals (librosa)

1. **Silence Ratio**
   - **What**: Percentage of video with silence or very low audio amplitude
   - **How**: RMS energy < threshold (e.g., -40dB) per frame
   - **Why**: Dead-air = viewer drop-off; high performers minimize silence
   - **Code location**: `extract_silence_ratio(video_path: str) -> float`

2. **Loudness Variance**
   - **What**: Standard deviation of RMS energy over time
   - **How**: Compute frame-wise RMS, aggregate variance
   - **Why**: Dynamic audio (varied speaking, music, SFX) indicates engagement
   - **Code location**: `extract_loudness_variance(video_path: str) -> float`

### Implementation Details

- **Frame sampling**: Extract frames at 10 fps (sufficient for detection, manageable file size)
- **Audio sampling**: Load audio at 22.05 kHz, compute frame-wise metrics
- **Documentation**: Each function must include docstring with:
  - What metric is extracted
  - Why it matters for viewer engagement
  - Trade-offs made (e.g., "using histogram diff vs optical flow for speed")

**Deliverable**: `app.py` with 5 signal extraction functions, each ~20–30 lines

---

## Phase 3: Analysis & Visualization

**Goal**: Compare high vs. low cohorts; identify statistically significant differences.

### Steps

1. **Process all 4 videos**:
   - For each video, call all signal extraction functions
   - Store results in pandas DataFrame (rows=videos, cols=signals)
   - Print summary statistics (mean, std, min, max)

2. **Comparative Analysis**:
   - Compute mean ± std for high-performing cohort
   - Compute mean ± std for low-performing cohort
   - Run t-test (or Mann-Whitney if small n) for each signal
   - Flag significant differences (p < 0.05)

3. **Visualizations** (keep simple):
   - Box plot: high vs. low for each signal (one subplot per signal)
   - Scatter plot: video rank vs. each metric
   - Table: summary stats for easy reference
   - Save as `output/signals_comparison.png`

4. **Sanity Checks**:
   - Plot metric values over time (e.g., scene cuts per minute)
   - Visually inspect: do extracted signals align with re-watching videos?

**Deliverable**: `output/signals_comparison.png` + printed statistics table

---

## Phase 4: Insight Generation

**Goal**: Convert metrics into 2–3 actionable, plain-language insights.

### Structure per Insight

```
**Signal**: [metric name]
- High performers: [mean ± std] (e.g., 8.2 ± 1.1 cuts/min)
- Low performers: [mean ± std] (e.g., 3.1 ± 0.8 cuts/min)
- Difference: [quantified gap]

**Insight**: "[Action high performers take]. [Action low performers avoid]. 
This likely drives engagement because [reasoning]."

**Recommendation**: [Specific, testable action for creator]
```

### Example

```
**Signal**: Scene Cut Frequency
- High performers: 8.2 ± 1.1 cuts/min
- Low performers: 3.1 ± 0.8 cuts/min

**Insight**: "High-performing streams introduce a visual change every 7–9 seconds, 
while low-performing streams remain static for 19–20 seconds. Frequent cuts sustain 
viewer attention and reduce cognitive load during passive consumption."

**Recommendation**: Aim for 1 scene change every 8–10 seconds through cuts, 
transitions, or on-screen movement.
```

### Validation

- **Re-watch videos** while tracking extracted metrics
- **Confirm pattern**: do high-performers actually have more cuts?
- **Sanity-check reasoning**: does the explanation make sense?

**Deliverable**: `INSIGHTS.md` (1-page summary with 2–3 insights)

---

## Phase 5: Deliverables

**Goal**: Package code, documentation, and insights for submission.

### 1. Python Code (`app.py`)

- Modular structure: import/download functions, signal extraction, analysis pipeline
- Main entry point: download videos → extract signals → compare → visualize
- Comments: trade-offs, assumptions, edge cases
- Example: `python app.py --videos videos.json --output output/`

### 2. README.md

Structure:

```markdown
# YouTube Video Analyzer

## Quick Start
- Prerequisites (Python 3.8+, pip install -r requirements.txt)
- How to run: python app.py --videos videos.json
- Output: signals_comparison.png, summary statistics

## Methodology
- Niche selected: [e.g., productivity vlogs]
- Signals extracted: [list of 5 signals]
- Videos analyzed: [2 high, 2 low with justification]

## Assumptions
- Consistent frame rate / audio sample rate
- No extreme outliers (e.g., 1-hour silence) that break metrics
- Scene cuts = frame histogram difference > threshold
- Silence = RMS < -40dB

## Limitations
- Cannot measure storytelling quality, production value, or personality
- Small sample (4 videos) → patterns observed, not generalized
- Signal thresholds tuned to this niche; may not transfer to others
- Manual validation required: re-watch videos to confirm metrics align

## Trade-offs Made
- Histogram-based cut detection (fast) vs optical flow (more accurate)
- Frame sampling at 10 fps (sufficient for this analysis) vs full framerate
- Simple metrics (std, mean) vs advanced statistical modeling
```

### 3. INSIGHTS.md

- 1-page summary of 2–3 key findings
- Format: Signal → Evidence → Actionable Recommendation
- Plain language, no jargon

### 4. Optional (Bonus)

- `requirements.txt`: exact package versions
- `output/`: plots and statistics
- `videos.json`: metadata on selected videos

**Deliverable**: All files committed; submit with CV to hiring contact

---

## Timeline & Milestones

| Phase | Estimated Time | Checkpoint |
|-------|----------------|-----------|
| Phase 1: Video Selection | 1–2 hours | List of 4 videos + justification |
| Phase 2: Signal Extraction | 4–6 hours | `app.py` with 5 functions working end-to-end |
| Phase 3: Analysis & Visualization | 2–3 hours | Plots generated, statistics printed |
| Phase 4: Insight Generation | 2–3 hours | `INSIGHTS.md` drafted, validated by re-watching |
| Phase 5: Documentation | 1–2 hours | README, comments, final review |
| **Total** | **10–16 hours** | **Ready to submit** |

---

## KISS Principles Applied

✅ **Off-the-shelf libraries**: OpenCV, librosa, pandas (no custom ML)
✅ **Simple metrics**: histogram diff, RMS, variance (interpretable, explainable)
✅ **No deep learning**: Avoids black boxes and high compute
✅ **4 videos**: Sufficient to observe patterns without massive data processing
✅ **Manual validation**: Insights grounded in actual video observation
✅ **Modular code**: Each signal is a standalone function, easy to test/modify

---

## Red Flags to Avoid

❌ Building a complex model before exploring the data
❌ Extracting signals without explaining business relevance
❌ Generating insights that contradict the metrics
❌ Over-generalizing from 4 videos to all YouTube content
❌ Spending time on deployment/infrastructure
❌ Using thresholds without tuning to your video data

---

## Success Criteria

- [x] 3–5 meaningful signals with clear rationale
- [x] Statistical comparison of high vs. low cohorts
- [x] 2–3 actionable, evidence-based insights in plain language
- [x] README explaining assumptions, limitations, how to run
- [x] Clean, understandable code with docstrings and comments
- (Optional) Simple dashboard or scoring model
