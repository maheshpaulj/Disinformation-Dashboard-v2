import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import spacy
from collections import Counter

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="Disinformation Ecosystem Dashboard",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS for a professional look ---
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #FFFFFF;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px #000000;
    }
    .metric-card {
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #4b6cb7;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        background-color: #f8f9fa;
    }
    .stMetric {
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #F0F2F6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF;
    }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# DATA LOADING & CACHING
# ==============================================================================
@st.cache_data
def load_data():
    try:
        df = pd.read_pickle('data/processed_data_with_features.pkl')
        cluster_df = pd.read_pickle('data/final_clusters_with_metadata.pkl')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        cluster_df['start_date'] = pd.to_datetime(cluster_df['start_date'])
        cluster_df = cluster_df.sort_values('size', ascending=False).reset_index(drop=True)
        return df, cluster_df
    except FileNotFoundError as e:
        st.error(f"ERROR: Data file not found: `{e.filename}`. Please ensure data files are in the correct subdirectories (`data/`, `results/networks/`).")
        return None, None

@st.cache_resource
def load_network_graph():
    try:
        return nx.read_gexf('results/networks/author_network.gexf')
    except FileNotFoundError:
        st.sidebar.warning("`author_network.gexf` not found. Network analysis will be limited.")
        return None

@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        st.sidebar.warning("SpaCy model not found. NER disabled. Run: `python -m spacy download en_core_web_sm`")
        return None

df, cluster_df = load_data()
nlp = load_spacy_model()
author_network = load_network_graph()

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
@st.cache_data
def get_top_entities(posts, _nlp_model, top_n=10):
    if _nlp_model is None or posts.empty: return {}
    text = " ".join(posts['text_clean'].head(100))
    doc = _nlp_model(text[:1000000])
    entities = Counter()
    for ent in doc.ents:
        if ent.label_ in ['PERSON', 'ORG', 'GPE']:
            entities[ent.text.strip().title()] += 1
    return dict(entities.most_common(top_n))

@st.cache_data
def generate_wordcloud(posts):
    text = ' '.join(posts['text_clean'].dropna())
    if not text: return None
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(text)
    return wordcloud.to_array()

# ==============================================================================
# MAIN DASHBOARD LAYOUT
# ==============================================================================

if df is None:
    st.stop()

st.markdown('<div class="main-header">Disinformation Ecosystem Dashboard</div>', unsafe_allow_html=True)

# --- SIDEBAR FILTERS ---
st.sidebar.title("🔬 Filters & Controls")
date_range = st.sidebar.slider("Date Range", df['timestamp'].min().date(), df['timestamp'].max().date(), (df['timestamp'].min().date(), df['timestamp'].max().date()))
selected_platforms = st.sidebar.multiselect("Platforms", df['source'].unique(), df['source'].unique())

filtered_df = df[(df['timestamp'].dt.date >= date_range[0]) & (df['timestamp'].dt.date <= date_range[1]) & (df['source'].isin(selected_platforms))]
valid_clusters = filtered_df[~filtered_df['cluster_global'].str.endswith('-1')]['cluster_global'].unique()
filtered_cluster_df = cluster_df[cluster_df['cluster_id'].isin(valid_clusters)].copy()

st.sidebar.info(f"Displaying **{len(filtered_df):,}** posts matching filters.")

# --- TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🎯 Narrative Explorer", "🕸️ Network Analysis", "⚙️ Methodology"])

# ==============================================================================
# TAB 1: OVERVIEW
# ==============================================================================
with tab1:
    st.header("Ecosystem at a Glance")
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.markdown(f"<div class='metric-card'> <h4>Total Posts</h4> <h1>{len(filtered_df):,}</h1> </div>", unsafe_allow_html=True)
    with col2: st.markdown(f"<div class='metric-card'> <h4>Narratives</h4> <h1>{len(filtered_cluster_df):,}</h1> </div>", unsafe_allow_html=True)
    with col3: st.markdown(f"<div class='metric-card'> <h4>Unique Authors</h4> <h1>{filtered_df['author'].nunique():,}</h1> </div>", unsafe_allow_html=True)
    with col4: st.markdown(f"<div class='metric-card'> <h4>Platforms</h4> <h1>{len(selected_platforms)}</h1> </div>", unsafe_allow_html=True)

    st.markdown("---")
    
    st.subheader("Activity Over Time")
    time_series_data = filtered_df.set_index('timestamp').resample('ME').size()
    st.plotly_chart(px.area(time_series_data, x=time_series_data.index, y=time_series_data.values, labels={'y': 'Posts', 'x': 'Date'}, title="Monthly Post Volume"), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Platform Distribution")
        platform_counts = filtered_df['source'].value_counts()
        st.plotly_chart(px.pie(platform_counts, values=platform_counts.values, names=platform_counts.index), use_container_width=True)
    with c2:
        st.subheader("Top 10 Narratives")
        st.dataframe(filtered_cluster_df[['cluster_id', 'size', 'duration_days']].head(10), use_container_width=True)

# ==============================================================================
# TAB 2: NARRATIVE EXPLORER
# ==============================================================================
with tab2:
    st.header("🎯 Narrative Explorer")
    if filtered_cluster_df.empty:
        st.warning("No narratives match the current filter settings.")
    else:
        selected_cluster_id = st.selectbox("Select a Narrative (sorted by size)", filtered_cluster_df['cluster_id'].tolist())
        cluster_details = filtered_cluster_df[filtered_cluster_df['cluster_id'] == selected_cluster_id].iloc[0]
        cluster_posts = filtered_df[filtered_df['cluster_global'] == selected_cluster_id]

        st.subheader(f"Analysis of Narrative: `{selected_cluster_id}`")
        c1, c2, c3 = st.columns(3)
        c1.metric("Size (Posts)", f"{cluster_details['size']:,}")
        c2.metric("Duration (Days)", f"{cluster_details['duration_days']:.0f}")
        c3.metric("Unique Authors", f"{cluster_details['unique_authors']:,}")

        avg_disinfo = cluster_posts['disinfo_marker_count'].mean()
        if avg_disinfo > 0.5: st.error(f"**High Disinfo Score:** {avg_disinfo:.2f} markers/post")
        elif avg_disinfo > 0.2: st.warning(f"**Moderate Disinfo Score:** {avg_disinfo:.2f} markers/post")
        else: st.success(f"**Low Disinfo Score:** {avg_disinfo:.2f} markers/post")

        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("**Platform Mix**")
            platform_mix = cluster_posts['source'].value_counts()
            fig_mix = go.Figure(data=[go.Pie(labels=platform_mix.index, values=platform_mix.values, hole=.5, textinfo='percent+label')])
            fig_mix.update_layout(showlegend=False, margin=dict(l=0, r=0, t=0, b=0), height=250)
            st.plotly_chart(fig_mix, use_container_width=True)
            
            st.markdown("**Top Entities**")
            top_entities = get_top_entities(cluster_posts, nlp)
            if top_entities: st.json(top_entities)
        with col2:
            st.markdown("**Word Cloud**")
            wordcloud_image = generate_wordcloud(cluster_posts)
            if wordcloud_image is not None:
                fig, ax = plt.subplots(); ax.imshow(wordcloud_image, interpolation='bilinear'); ax.axis("off"); st.pyplot(fig)
        
        st.subheader("Top Authors & Sample Posts")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Top 10 Authors in this Narrative**")
            st.dataframe(cluster_posts['author'].value_counts().head(10))
        with c2:
            st.markdown("**Most Engaged Posts**")
            st.dataframe(cluster_posts[['text', 'interactions']].nlargest(5, 'interactions'), hide_index=True)

# ==============================================================================
# TAB 3: NETWORK ANALYSIS
# ==============================================================================
with tab3:
    st.header("🕸️ Network Analysis")
    if author_network is None:
        st.error("Author network file not loaded. Cannot perform network analysis.")
    else:
        nodes_in_filtered_df = set(filtered_df['author'].unique())
        subgraph_nodes = [node for node in author_network.nodes() if node in nodes_in_filtered_df]
        G = author_network.subgraph(subgraph_nodes).copy()

        if G.number_of_nodes() < 2:
            st.warning("Not enough author connections to build a network with the current filters.")
        else:
            largest_cc_nodes = max(nx.connected_components(G), key=len)
            G_largest = G.subgraph(largest_cc_nodes).copy()
            st.markdown(f"Analyzing the largest connected component of the network with **{G_largest.number_of_nodes():,}** authors and **{G_largest.number_of_edges():,}** connections.")

            c1, c2, c3 = st.columns(3)
            c1.metric("Network Density", f"{nx.density(G_largest):.4f}")
            c2.metric("Avg. Connections", f"{np.mean([d for n, d in G_largest.degree()]):.2f}")
            c3.metric("Communities", G_largest.graph.get('__modularity', 'N/A')) # Louvain is pre-calculated in notebook

            st.subheader("Top 20 Influential Authors")
            degree = pd.Series(dict(G_largest.degree(weight='weight'))).sort_values(ascending=False)
            betweenness = pd.Series(nx.betweenness_centrality(G_largest, k=min(100, G_largest.number_of_nodes()))).sort_values(ascending=False)
            influencer_df = pd.DataFrame({'Degree (Connections)': degree, 'Betweenness (Broker Role)': betweenness}).head(20)
            st.dataframe(influencer_df, use_container_width=True)

            st.subheader("Interactive Network Graph")
            render_graph = st.checkbox("Render interactive graph (slow for > 500 nodes)", value=G_largest.number_of_nodes() <= 500)
            if render_graph:
                with st.spinner("Calculating graph layout..."):
                    pos = nx.spring_layout(G_largest, seed=42)
                    edge_x, edge_y = [], []
                    for edge in G_largest.edges():
                        x0, y0 = pos[edge[0]]; x1, y1 = pos[edge[1]]
                        edge_x.extend([x0, x1, None]); edge_y.extend([y0, y1, None])
                    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')
                    node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
                    for node in G_largest.nodes():
                        x, y = pos[node]; node_x.append(x); node_y.append(y)
                        degree = G_largest.degree(node); community = G_largest.nodes[node].get('community', 0)
                        node_size.append(5 + degree); node_color.append(community)
                        node_text.append(f"Author: {node}<br>Community: {community}<br>Connections: {degree}")
                    node_trace = go.Scatter(x=node_x, y=node_y, mode='markers', hoverinfo='text', text=node_text,
                                            marker=dict(size=node_size, color=node_color, colorscale='Viridis', showscale=True,
                                                        colorbar=dict(title='Community ID')))
                    fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(showlegend=False, margin=dict(b=0,l=0,r=0,t=40), height=700))
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Graph rendering disabled due to size. Use Gephi for large network visualization.")

# ==============================================================================
# TAB 4: METHODOLOGY
# ==============================================================================
with tab4:
    st.header("⚙️ Methodology Explained")
    st.markdown("This dashboard is the result of a multi-stage data pipeline designed to map and analyze disinformation ecosystems.")
    
    with st.expander("### 1. Data Collection", expanded=True):
        st.markdown("- **Multi-Source Aggregation:** Data was collected from diverse platforms including Reddit, Telegram, 4chan, and News APIs to capture a wide spectrum of online discourse.")
        st.markdown("- **Keyword-Driven Targeting:** The collection was guided by a curated list of keywords and phrases related to prominent disinformation themes (e.g., vaccine misinformation, election fraud, climate denial).")

    with st.expander("### 2. Feature Engineering"):
        st.markdown("""
        To understand the nuances of each post, we calculated various features:
        - **Semantic Features (Embeddings):** The text of each post was converted into a 384-dimensional vector using the `all-MiniLM-L6-v2` sentence transformer model. These vectors capture the *meaning* of the text, allowing us to group posts that are semantically similar even if they don't use the exact same words.
        - **Linguistic & Marker Features:** We analyzed writing style (length, capitalization) and scanned for specific phrases commonly associated with disinformation tactics (e.g., "do your own research," "deep state").
        """)

    with st.expander("### 3. Time-Aware Adaptive Clustering (HDBSCAN)"):
        st.markdown("""
        This is the core of our narrative detection. The goal is to group similar posts into "narrative clusters."
        - **Time-Aware:** The data is first sliced into 2-month time windows. Clustering is performed independently within each window. This allows us to track how narratives evolve, appear, and fade over time, rather than treating a decade of data as a single static block.
        - **Adaptive Clustering (HDBSCAN):** We chose the **HDBSCAN** algorithm for two key reasons:
            1.  **It finds clusters of varying shapes and densities.** Real-world narratives aren't uniform. HDBSCAN can identify both small, tightly-coordinated message campaigns and larger, more diffuse conversations.
            2.  **It automatically identifies noise.** HDBSCAN excels at labeling posts that don't belong to any coherent group as "noise." This is crucial for filtering out random chatter and focusing our analysis only on true, cohesive narratives. This is why you see a "Noise Percentage" in the analysis—it's a feature, not a bug!
        """)

    with st.expander("### 4. Network Analysis"):
        st.markdown("- **Author Co-occurrence Network:** We built a graph where each node is an author. An edge connects two authors if they both participated in the same narrative cluster. The more narratives they share, the stronger the connection.")
        st.markdown("- **Community Detection & Influencer Analysis:** Using the Louvain algorithm and centrality metrics on this network, we can identify influential authors and "echo chambers"—groups of authors who are more densely connected to each other than to the rest of the network.")