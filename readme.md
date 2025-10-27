# 🔍 Time-Aware Multi-Platform Disinformation Ecosystem Mapping

**Complete Research Pipeline with Interactive Dashboard**

---

## 📋 Project Overview

This project implements a comprehensive pipeline for collecting, analyzing, and visualizing disinformation narratives across multiple social media platforms. It includes:

- **Multi-source data collection** (Reddit, Telegram, 4chan, News)
- **Advanced NLP feature engineering**
- **Time-aware adaptive clustering**
- **Network analysis** (influencers, echo chambers, cross-platform flows)
- **Interactive Streamlit dashboard**

**Expected Output**: 50,000-150,000 posts with rich metadata, cluster analysis, and network visualizations.

---

## 🗂️ Project Structure

```
fp2_project/
│
├── Part 1A: Reddit + NewsAPI Collection
│   ├── reddit_news_collection.py
│   └── partial_data_reddit_news/
│
├── Part 1B: Telegram + 4chan Collection  
│   ├── telegram_4chan_collection.py
│   └── partial_data_telegram_4chan/
│
├── Combine Datasets
│   └── combine_datasets.py
│
├── Part 2: Feature Engineering & Clustering
│   └── feature_engineering_clustering.py
│
├── Part 3: Network Analysis
│   └── network_analysis_visualization.py
│
├── Part 4: Interactive Dashboard
│   ├── dashboard_app.py
│   ├── quick_setup.py
│   └── requirements.txt
│
├── Generated Data Files
│   ├── reddit_news_data.pkl
│   ├── telegram_4chan_data.pkl
│   ├── unified_raw_data_MASSIVE_v2.pkl
│   ├── processed_data_with_features.pkl
│   └── final_clusters_with_metadata.pkl
│
└── Results & Visualizations
    ├── networks/
    │   ├── author_network.gexf
    │   ├── cluster_network.gexf
    │   ├── interactive_network.html
    │   └── platform_flow_sankey.html
    ├── dashboard.html
    └── FINAL_REPORT.txt
```

---

## 🚀 Quick Start (3 Steps)

### 1️⃣ Get API Keys (10 minutes)

Follow the **API Keys Setup Guide** to get:
- ✅ Reddit API (free, 2 min)
- ✅ NewsAPI (free, 2 min)  
- ✅ Telegram API (free, 5 min)
- ✅ 4chan - NO KEY NEEDED!

### 2️⃣ Collect & Process Data (3-5 hours compute)

**Option A: Sequential (Easier)**
```python
# Step 1: Collect Reddit + News (30-60 min)
# Run Part 1A in Google Colab

# Step 2: Collect Telegram + 4chan (45-90 min)
# Run Part 1B in Google Colab

# Step 3: Combine datasets (2 min)
# Run combine_datasets.py

# Step 4: Feature engineering (20-40 min)
# Run Part 2

# Step 5: Network analysis (15-30 min)
# Run Part 3
```

**Option B: Parallel (Faster - 2-3 hours)**
```python
# Open 2 Colab notebooks simultaneously
# Notebook 1: Run Part 1A
# Notebook 2: Run Part 1B
# Then combine and continue with Parts 2-3
```

### 3️⃣ Launch Dashboard (2 minutes)

```bash
# Download data files from Google Drive to your computer
# Save dashboard_app.py in same folder

pip install streamlit plotly networkx pandas numpy
streamlit run dashboard_app.py
```

**🎉 Done!** Dashboard opens at `http://localhost:8501`

---

## 📊 What You'll Get

### Data Collection Results
- **50,000-150,000 posts** across 4-6 platforms
- **6-20 narrative clusters** with temporal tracking
- **Network graphs** (Gephi-ready)
- **Comprehensive metadata** (engagement, markers, etc.)

### Visualizations
- 15+ static charts (PNG/PDF)
- 10+ interactive HTML visualizations
- Network graph files (.gexf, .graphml)
- Comprehensive dashboard with 6 analysis tabs

### Academic Outputs
- Final analysis report (TXT)
- Cluster metadata (CSV)
- Key influencers list (CSV)
- Platform flow analysis (CSV)
- Sample datasets for review

---

## 🎯 Dashboard Features

### Tab 1: Overview 📊
- Dataset statistics & key metrics
- Platform distribution
- Activity timeline
- Top authors analysis
- Engagement patterns

### Tab 2: Network Analysis 🕸️
- Interactive co-occurrence network
- Centrality metrics
- Community detection
- Influencer identification

### Tab 3: Temporal Analysis 📈
- Time-series visualization
- Day/hour activity patterns
- Platform evolution
- Trend analysis

### Tab 4: Cluster Explorer 🎯
- Cluster size distribution
- Emergence timeline
- Individual cluster deep-dive
- Sample post viewer

### Tab 5: Disinformation Analysis ⚠️
- Marker prevalence tracking
- Co-occurrence patterns
- Temporal evolution
- Statistical summaries

### Tab 6: Search & Investigate 🔍
- Full-text search
- Author profiles
- CSV export
- Custom filtering

---

## 🛠️ Technical Stack

### Data Collection
- `praw` - Reddit API wrapper
- `telethon` - Telegram API
- `newsapi-python` - News aggregation
- `requests` + `BeautifulSoup` - 4chan scraping

### Analysis & Processing
- `pandas` - Data manipulation
- `scikit-learn` - ML & clustering
- `sentence-transformers` - Embeddings
- `networkx` - Graph analysis

### Visualization
- `streamlit` - Interactive dashboard
- `plotly` - Interactive charts
- `matplotlib` + `seaborn` - Static plots
- `wordcloud` - Text visualization

---

## 📈 Performance Specs

### Compute Requirements
- **CPU**: Modern multi-core (4+ cores recommended)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 2-5GB for data + results
- **Time**: 3-5 hours total runtime

### Data Scale
| Metric | Expected Range |
|--------|---------------|
| Total Posts | 50K - 150K |
| Unique Authors | 10K - 50K |
| Clusters | 20 - 100 |
| Network Edges | 1K - 50K |
| Time Span | 3-5 years |

---

## 🔧 Customization Options

### Add New Platforms

```python
# In Part 1, add new scraping function
def scrape_new_platform(query, limit):
    # Your scraping logic
    return pd.DataFrame(posts)

# Add to collection loop
for keyword in config['keywords']:
    df = scrape_new_platform(keyword, 1000)
    # Save as pickle
```

### Add New Themes

```python
THEMES = {
    "Your_New_Theme": {
        "keywords": ['"keyword1"', '"keyword2"'],
        "subreddits": ['subreddit1', 'subreddit2'],
        "telegram_channels": ['s/channel1'],
        "4chan_boards": ['pol']
    }
}
```

### Modify Clustering

```python
# In Part 2, adjust DBSCAN parameters
eps = 0.5  # Lower = tighter clusters
min_samples = 10  # Higher = larger minimum cluster size
```

### Custom Dashboard Metrics

```python
# In dashboard_app.py, add new metrics
st.metric("Your Metric", f"{your_calculation}")
```

---

## 📝 For Your Academic Report

### Methodology Section
```
Data was collected using authenticated APIs for Reddit (PRAW) and Telegram,
public archives for 4chan (4plebs.org), and NewsAPI for mainstream coverage.
Collection spanned [DATE_RANGE], resulting in [N] posts across [X] platforms.

Feature engineering employed SBERT embeddings (all-MiniLM-L6-v2) for semantic
representation, combined with 40+ linguistic, temporal, and behavioral features.

Time-aware clustering used DBSCAN with 2-month temporal windows to capture
evolving narratives. Network analysis identified influencers via degree and
betweenness centrality metrics.
```

### Results Section
```
The analysis identified [N] distinct narrative clusters, with the largest
containing [X] posts spanning [Y] days. Cross-platform analysis revealed
[FINDING] with [PLATFORM A] → [PLATFORM B] being the dominant flow pattern.

Top influencers were responsible for [X]% of content, suggesting centralized
information dissemination. Disinformation markers appeared in [Y]% of posts,
with '[TOP_MARKER]' being the most prevalent pattern.
```

### Figures to Include
1. Platform distribution (pie chart)
2. Network visualization (Gephi export)
3. Cluster timeline (scatter plot)
4. Disinformation markers (bar chart)
5. Temporal evolution (line chart)

---

## 🐛 Troubleshooting

### "API Rate Limit Exceeded"
→ Wait 15 minutes, scripts auto-resume from partial data

### "Out of Memory" Error
→ Reduce sample size or process in smaller batches

### "No Module Named X"
→ `pip install -r requirements.txt`

### Dashboard Won't Load
→ Check file paths are correct (absolute paths work best)

### Charts Not Showing
→ Update Plotly: `pip install --upgrade plotly`

---

## 📚 Citation

If you use this pipeline in your research, please cite:

```bibtex
@misc{fp2_disinfo_pipeline,
  author = {Your Name},
  title = {Time-Aware Multi-Platform Disinformation Ecosystem Mapping},
  year = {2024},
  url = {https://github.com/yourusername/fp2-project}
}
```

---

## 📄 License

This project is for academic research purposes. Data collected is subject to
platform terms of service. Use responsibly and ethically.

---

## 🆘 Support

### Common Issues
- Check the troubleshooting guide
- Review error messages carefully
- Ensure all API keys are set correctly

### Resources
- Streamlit Docs: https://docs.streamlit.io
- PRAW Docs: https://praw.readthedocs.io
- NetworkX Docs: https://networkx.org

---

## ✅ Success Checklist

- [ ] API keys obtained and tested
- [ ] Part 1A completed (Reddit + News)
- [ ] Part 1B completed (Telegram + 4chan)
- [ ] Datasets combined successfully
- [ ] Part 2 completed (features + clustering)
-