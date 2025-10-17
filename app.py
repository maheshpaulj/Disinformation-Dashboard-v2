import streamlit as st
import pandas as pd
import plotly.express as px

# --- PAGE CONFIGURATION ---
# Use st.set_page_config() as the first Streamlit command.
st.set_page_config(
    page_title="Disinformation Ecosystem Mapper",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/maheshpaulj/Disinformation-Dashboard-v2/', # Replace with your GitHub repo
        'Report a bug': "https://github.com/maheshpaulj/Disinformation-Dashboard-v2/issues", # Replace
        'About': """
        ## Disinformation Ecosystem Mapper
        This dashboard provides an interactive tool to explore and analyze online narratives across multiple platforms. 
        It uses unsupervised machine learning to cluster conversations and identify key themes and actors.
        """
    }
)

# --- DATA LOADING ---
# Use Streamlit's caching to load data only once, improving performance.
@st.cache_data
def load_data():
    """Loads the final processed data from a pickle file."""
    try:
        df = pd.read_pickle('data/dashboard_data_final.pkl')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        # Ensure numeric columns are treated correctly, filling NaNs where necessary
        numeric_cols = ['sentiment', 'toxicity', 'interactions', 'user_followers']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        return df
    except FileNotFoundError:
        st.error("Data file not found. Please make sure `data/dashboard_data_final.pkl` exists.")
        return None

df = load_data()

# Stop the app if data loading fails
if df is None:
    st.stop()

# Get the list of auto-generated narrative labels for the filter
narrative_labels = sorted([label for label in df['narrative_label'].unique() if label != "Noise / Uncategorized"])


# --- SIDEBAR & FILTERS ---
with st.sidebar:
    st.title("Discovery Controls")
    st.markdown("Use the filters below to explore the narrative ecosystem.")

    # Date Range Slider
    min_date = df['timestamp'].min().date()
    max_date = df['timestamp'].max().date()
    selected_date_range = st.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        help="Filter the data to a specific time window."
    )

    # Platform Selector
    platform_options = sorted(df['source'].unique())
    selected_platforms = st.multiselect(
        "Select Platforms",
        options=platform_options,
        default=platform_options,
        help="Include or exclude data from specific online platforms."
    )

    # Narrative Selector for the deep dive
    selected_narrative = st.selectbox(
        "Select a Narrative to Analyze",
        options=narrative_labels,
        help="Choose a specific narrative cluster to see its detailed breakdown."
    )
    
    st.info("The **Ecosystem Map** shows all data based on your filters. The **Narrative Deep Dive** focuses only on the single narrative selected above.")


# --- DATA FILTERING LOGIC ---
# Apply filters based on sidebar selections
start_date, end_date = pd.to_datetime(selected_date_range[0]), pd.to_datetime(selected_date_range[1])
filtered_df = df[
    (df['timestamp'].dt.date >= start_date.date()) &
    (df['timestamp'].dt.date <= end_date.date()) &
    (df['source'].isin(selected_platforms))
]


# --- MAIN PAGE LAYOUT ---
st.title("Disinformation Ecosystem Mapper")
st.markdown("An interactive dashboard to identify and analyze online narratives across multiple platforms.")
st.markdown("---")

# --- MAIN TABS ---
tab1, tab2 = st.tabs(["🗺️ Ecosystem Map", f"🔬 Narrative Deep Dive"])

# --- TAB 1: ECOSYSTEM MAP ---
with tab1:
    st.header("Map of All Filtered Narratives")
    st.markdown("""
    This is a "map of ideas". Each dot represents a post.
    - **Proximity**: Posts that are close together are semantically similar in meaning.
    - **Color**: Posts with the same color belong to the same narrative cluster.
    - **Interaction**: Hover over any point to see the post's text and source.
    """)
    
    if not filtered_df.empty:
        fig_scatter = px.scatter(
            filtered_df,
            x='umap_x',
            y='umap_y',
            color='narrative_label',
            hover_name='narrative_label',
            hover_data={
                'text': True,
                'source': True,
                'umap_x': False, # Hide coordinates from hover
                'umap_y': False
            },
            color_discrete_sequence=px.colors.qualitative.Plotly,
            title="UMAP Projection of Narrative Clusters"
        )
        fig_scatter.update_layout(legend_title_text='Narrative Clusters')
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.warning("No data available for the selected filters in the Ecosystem Map.")

# --- TAB 2: NARRATIVE DEEP DIVE ---
with tab2:
    st.header(f"In-Depth Analysis of: \"{selected_narrative}\"")
    
    # Isolate data for the selected narrative from the filtered data
    narrative_df = filtered_df[filtered_df['narrative_label'] == selected_narrative]
    
    if narrative_df.empty:
        st.warning("No data available for this narrative in the selected date range or platforms.")
    else:
        # --- KEY METRICS FOR THIS NARRATIVE ---
        st.subheader("Narrative Profile")
        avg_sentiment = narrative_df['sentiment'].mean()
        avg_toxicity = narrative_df['toxicity'].mean()
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Posts in View", f"{len(narrative_df):,}")
        col2.metric(
            "Average Sentiment",
            f"{avg_sentiment:.2f}",
            help="A score from -1 (very negative) to +1 (very positive). Near-zero is neutral."
        )
        col3.metric(
            "Average Toxicity Score",
            f"{avg_toxicity:.2f}",
            help="A score from 0 to 1 indicating the model's confidence that the content is toxic."
        )

        st.markdown("---")

        # --- EVOLUTION TIMELINE ---
        st.subheader("Narrative Timeline")
        time_series = narrative_df.set_index('timestamp').resample('D').size().reset_index(name='count')
        fig_line = px.line(
            time_series,
            x='timestamp',
            y='count',
            title="Daily Post Volume for this Narrative",
            labels={'count': 'Number of Posts', 'timestamp': 'Date'}
        )
        st.plotly_chart(fig_line, use_container_width=True)

        # --- TOP PLATFORMS AND ACTORS ---
        col_plat, col_actor = st.columns(2)
        with col_plat:
            st.subheader("Top Platforms")
            platform_dist = narrative_df['source'].value_counts().reset_index()
            st.dataframe(platform_dist)
        with col_actor:
            st.subheader("Top Posters (by frequency)")
            actor_dist = narrative_df['author'].value_counts().nlargest(10).reset_index()
            st.dataframe(actor_dist)
            
        # --- SAMPLE POSTS ---
        st.subheader("Sample Posts from this Narrative")
        st.dataframe(
            narrative_df[['text', 'source', 'timestamp', 'sentiment', 'toxicity', 'interactions']]
            .sort_values(by='interactions', ascending=False)
            .head(15)
        )