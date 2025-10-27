# ==============================================================================
# FP-2 PROJECT: PART 4 - INTERACTIVE STREAMLIT DASHBOARD
# ==============================================================================
# Save this file as: dashboard_app.py
# Run with: streamlit run dashboard_app.py
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from datetime import datetime, timedelta
import pickle
from collections import Counter
import json

# Page configuration
st.set_page_config(
    page_title="Disinformation Ecosystem Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .insight-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# DATA LOADING FUNCTIONS
# ==============================================================================

@st.cache_data
def load_processed_data(file_path):
    """Load processed data with embeddings and features."""
    try:
        df = pd.read_pickle(file_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_data
def load_cluster_metadata(file_path):
    """Load cluster analysis results."""
    try:
        df = pd.read_pickle(file_path)
        return df
    except Exception as e:
        st.error(f"Error loading clusters: {e}")
        return None

@st.cache_data
def load_network_data(file_path):
    """Load network analysis results."""
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading network: {e}")
        return None

# ==============================================================================
# MAIN DASHBOARD
# ==============================================================================

def main():
    # Header
    st.markdown('<div class="main-header">🔍 Time-Aware Multi-Platform Disinformation Ecosystem Dashboard</div>', 
                unsafe_allow_html=True)
    
    # Sidebar - Data Loading
    st.sidebar.title("📁 Data Configuration")
    
    # File paths
    data_path = st.sidebar.text_input(
        "Processed Data Path",
        value="processed_data_with_features.pkl",
        help="Path to processed dataset from Part 2"
    )
    
    cluster_path = st.sidebar.text_input(
        "Cluster Metadata Path",
        value="final_clusters_with_metadata.pkl",
        help="Path to cluster metadata from Part 2"
    )
    
    # Load data
    if st.sidebar.button("🔄 Load Data", type="primary"):
        with st.spinner("Loading data..."):
            st.session_state['df'] = load_processed_data(data_path)
            st.session_state['clusters'] = load_cluster_metadata(cluster_path)
        
        if st.session_state['df'] is not None:
            st.sidebar.success("✅ Data loaded successfully!")
    
    # Check if data is loaded
    if 'df' not in st.session_state or st.session_state['df'] is None:
        st.info("👈 Please load data from the sidebar to begin analysis")
        st.stop()
    
    df = st.session_state['df']
    clusters = st.session_state['clusters'] if 'clusters' in st.session_state else None
    
    # Sidebar - Filters
    st.sidebar.title("🎯 Filters")
    
    # Date range filter
    min_date = df['timestamp'].min().date()
    max_date = df['timestamp'].max().date()
    
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Platform filter
    platforms = st.sidebar.multiselect(
        "Platforms",
        options=df['source'].unique(),
        default=df['source'].unique()
    )
    
    # Apply filters
    mask = (df['timestamp'].dt.date >= date_range[0]) & \
           (df['timestamp'].dt.date <= date_range[1]) & \
           (df['source'].isin(platforms))
    
    df_filtered = df[mask]
    
    st.sidebar.metric("Filtered Posts", f"{len(df_filtered):,}")
    
    # ==============================================================================
    # TAB NAVIGATION
    # ==============================================================================
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview",
        "🕸️ Network Analysis", 
        "📈 Temporal Analysis",
        "🎯 Cluster Explorer",
        "⚠️ Disinformation Analysis",
        "🔍 Search & Investigate"
    ])
    
    # ==============================================================================
    # TAB 1: OVERVIEW
    # ==============================================================================
    
    with tab1:
        st.header("📊 Dataset Overview")
        
        # Key metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Posts", f"{len(df_filtered):,}")
        
        with col2:
            st.metric("Unique Authors", f"{df_filtered['author'].nunique():,}")
        
        with col3:
            st.metric("Platforms", f"{df_filtered['source'].nunique()}")
        
        with col4:
            if clusters is not None:
                st.metric("Narrative Clusters", f"{clusters.shape[0]:,}")
            else:
                st.metric("Time Span", f"{(df_filtered['timestamp'].max() - df_filtered['timestamp'].min()).days} days")
        
        with col5:
            st.metric("Total Engagement", f"{df_filtered['interactions'].sum():,}")
        
        st.markdown("---")
        
        # Platform distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Platform Distribution")
            platform_counts = df_filtered['source'].value_counts()
            
            fig = px.pie(
                values=platform_counts.values,
                names=platform_counts.index,
                title="Posts by Platform",
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Activity Timeline")
            
            # Resample by day
            daily_posts = df_filtered.set_index('timestamp').resample('D').size()
            
            fig = px.line(
                x=daily_posts.index,
                y=daily_posts.values,
                title="Daily Post Volume",
                labels={'x': 'Date', 'y': 'Number of Posts'}
            )
            fig.update_traces(line_color='#667eea', line_width=2)
            fig.update_layout(hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)
        
        # Top authors
        st.subheader("🏆 Top 10 Most Active Authors")
        top_authors = df_filtered['author'].value_counts().head(10).reset_index()
        top_authors.columns = ['Author', 'Posts']
        top_authors['Avg Engagement'] = top_authors['Author'].map(
            df_filtered.groupby('author')['interactions'].mean()
        ).round(1)
        
        fig = px.bar(
            top_authors,
            x='Posts',
            y='Author',
            orientation='h',
            title="Most Active Authors",
            color='Avg Engagement',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Engagement distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Engagement Distribution")
            fig = px.histogram(
                df_filtered,
                x='interactions',
                nbins=50,
                title="Post Engagement Distribution",
                labels={'interactions': 'Interactions', 'count': 'Number of Posts'}
            )
            fig.update_traces(marker_color='#764ba2')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Content Statistics")
            
            if 'text_length' in df_filtered.columns:
                avg_length = df_filtered['text_length'].mean()
            else:
                avg_length = df_filtered['text'].str.len().mean()
            
            metrics_data = {
                'Metric': ['Avg Post Length', 'Avg Words/Post', 'Posts with URLs', 'Posts with Hashtags'],
                'Value': [
                    f"{avg_length:.0f} chars",
                    f"{df_filtered['text'].str.split().str.len().mean():.0f}",
                    f"{df_filtered['text'].str.contains('http').sum():,}" if 'has_url' not in df_filtered.columns else f"{df_filtered['has_url'].sum():,}",
                    f"{df_filtered['text'].str.contains('#').sum():,}" if 'hashtag_count' not in df_filtered.columns else f"{(df_filtered['hashtag_count'] > 0).sum():,}"
                ]
            }
            st.dataframe(pd.DataFrame(metrics_data), hide_index=True, use_container_width=True)
    
    # ==============================================================================
    # TAB 2: NETWORK ANALYSIS
    # ==============================================================================
    
    with tab2:
        st.header("🕸️ Network Analysis")
        
        if clusters is None:
            st.warning("⚠️ Cluster data not loaded. Some features may be limited.")
        
        st.subheader("Author Co-occurrence Network")
        
        # Build simple co-occurrence network
        with st.spinner("Building network..."):
            # Sample for performance
            sample_size = min(10000, len(df_filtered))
            df_sample = df_filtered.sample(n=sample_size, random_state=42)
            
            # Create author-author edges based on cluster participation
            if 'cluster_global' in df_sample.columns:
                edges = []
                for cluster in df_sample['cluster_global'].unique():
                    authors = df_sample[df_sample['cluster_global'] == cluster]['author'].unique()
                    if len(authors) > 1:
                        for i, a1 in enumerate(authors):
                            for a2 in authors[i+1:]:
                                edges.append((a1, a2))
                
                edge_counts = Counter(edges)
                
                # Filter to significant connections
                significant_edges = [(a1, a2, w) for (a1, a2), w in edge_counts.items() if w >= 3]
                
                st.metric("Network Edges", f"{len(significant_edges):,}")
                st.metric("Connected Authors", f"{len(set([e[0] for e in significant_edges] + [e[1] for e in significant_edges])):,}")
                
                # Create network visualization
                G = nx.Graph()
                for a1, a2, weight in significant_edges[:500]:  # Limit for visualization
                    G.add_edge(a1, a2, weight=weight)
                
                if G.number_of_nodes() > 0:
                    # Calculate network metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Network Density", f"{nx.density(G):.4f}")
                    
                    with col2:
                        st.metric("Connected Components", nx.number_connected_components(G))
                    
                    with col3:
                        degrees = [d for n, d in G.degree()]
                        st.metric("Avg Connections", f"{np.mean(degrees):.1f}")
                    
                    # Network visualization using plotly
                    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
                    
                    edge_x = []
                    edge_y = []
                    for edge in G.edges():
                        x0, y0 = pos[edge[0]]
                        x1, y1 = pos[edge[1]]
                        edge_x.extend([x0, x1, None])
                        edge_y.extend([y0, y1, None])
                    
                    edge_trace = go.Scatter(
                        x=edge_x, y=edge_y,
                        line=dict(width=0.5, color='#888'),
                        hoverinfo='none',
                        mode='lines'
                    )
                    
                    node_x = []
                    node_y = []
                    node_text = []
                    node_size = []
                    
                    for node in G.nodes():
                        x, y = pos[node]
                        node_x.append(x)
                        node_y.append(y)
                        degree = G.degree(node)
                        node_text.append(f"{node}<br>Connections: {degree}")
                        node_size.append(max(5, degree * 2))
                    
                    node_trace = go.Scatter(
                        x=node_x, y=node_y,
                        mode='markers',
                        hoverinfo='text',
                        text=node_text,
                        marker=dict(
                            showscale=True,
                            colorscale='Viridis',
                            size=node_size,
                            color=[G.degree(n) for n in G.nodes()],
                            colorbar=dict(title="Connections"),
                            line=dict(width=1, color='white')
                        )
                    )
                    
                    fig = go.Figure(data=[edge_trace, node_trace],
                                  layout=go.Layout(
                                      title="Author Co-occurrence Network",
                                      showlegend=False,
                                      hovermode='closest',
                                      margin=dict(b=0,l=0,r=0,t=40),
                                      xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                      height=600
                                  ))
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Most central authors
                    st.subheader("🎯 Most Central Authors")
                    
                    centrality = nx.degree_centrality(G)
                    top_central = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]
                    
                    central_df = pd.DataFrame(top_central, columns=['Author', 'Centrality Score'])
                    central_df['Centrality Score'] = central_df['Centrality Score'].round(3)
                    
                    st.dataframe(central_df, hide_index=True, use_container_width=True)
            
            else:
                st.info("Cluster information not available for network analysis")
    
    # ==============================================================================
    # TAB 3: TEMPORAL ANALYSIS
    # ==============================================================================
    
    with tab3:
        st.header("📈 Temporal Analysis")
        
        st.subheader("Activity Patterns Over Time")
        
        # Time aggregation selection
        time_agg = st.selectbox(
            "Time Aggregation",
            options=['Day', 'Week', 'Month'],
            index=1
        )
        
        freq_map = {'Day': 'D', 'Week': 'W', 'Month': 'M'}
        
        # Aggregate data
        temporal_data = df_filtered.set_index('timestamp').resample(freq_map[time_agg]).agg({
            'id': 'count',
            'interactions': 'sum',
            'author': 'nunique'
        }).reset_index()
        
        temporal_data.columns = ['Date', 'Posts', 'Total Engagement', 'Unique Authors']
        
        # Multi-line chart
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Post Volume', 'Total Engagement', 'Active Authors'),
            vertical_spacing=0.1
        )
        
        fig.add_trace(
            go.Scatter(x=temporal_data['Date'], y=temporal_data['Posts'],
                      mode='lines', name='Posts', line=dict(color='#667eea', width=2)),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=temporal_data['Date'], y=temporal_data['Total Engagement'],
                      mode='lines', name='Engagement', line=dict(color='#764ba2', width=2)),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=temporal_data['Date'], y=temporal_data['Unique Authors'],
                      mode='lines', name='Authors', line=dict(color='#f093fb', width=2)),
            row=3, col=1
        )
        
        fig.update_layout(height=800, showlegend=False, hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
        
        # Day of week / Hour patterns
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Activity by Day of Week")
            dow = df_filtered['timestamp'].dt.day_name()
            dow_counts = dow.value_counts().reindex([
                'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
            ])
            
            fig = px.bar(
                x=dow_counts.index,
                y=dow_counts.values,
                labels={'x': 'Day', 'y': 'Posts'},
                color=dow_counts.values,
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Activity by Hour")
            hour_counts = df_filtered['timestamp'].dt.hour.value_counts().sort_index()
            
            fig = px.line(
                x=hour_counts.index,
                y=hour_counts.values,
                labels={'x': 'Hour of Day', 'y': 'Posts'},
                markers=True
            )
            fig.update_traces(line_color='#667eea', line_width=3)
            st.plotly_chart(fig, use_container_width=True)
        
        # Platform evolution
        st.subheader("Platform Activity Evolution")
        
        platform_temporal = df_filtered.groupby([
            pd.Grouper(key='timestamp', freq=freq_map[time_agg]),
            'source'
        ]).size().reset_index(name='posts')
        
        fig = px.area(
            platform_temporal,
            x='timestamp',
            y='posts',
            color='source',
            title=f"Platform Activity Over Time ({time_agg}ly)",
            labels={'timestamp': 'Date', 'posts': 'Number of Posts'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # ==============================================================================
    # TAB 4: CLUSTER EXPLORER
    # ==============================================================================
    
    with tab4:
        st.header("🎯 Narrative Cluster Explorer")
        
        if clusters is None or 'cluster_global' not in df_filtered.columns:
            st.warning("⚠️ Cluster data not available")
        else:
            # Cluster overview
            st.subheader("Cluster Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Clusters", f"{clusters.shape[0]:,}")
            
            with col2:
                st.metric("Largest Cluster", f"{clusters['size'].max():,} posts")
            
            with col3:
                st.metric("Avg Duration", f"{clusters['duration_days'].mean():.1f} days")
            
            with col4:
                st.metric("Avg Cluster Size", f"{clusters['size'].mean():.0f} posts")
            
            # Cluster size distribution
            st.subheader("Cluster Size Distribution")
            
            fig = px.histogram(
                clusters,
                x='size',
                nbins=50,
                title="Distribution of Cluster Sizes",
                labels={'size': 'Cluster Size (posts)', 'count': 'Number of Clusters'},
                log_y=True
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Cluster timeline
            st.subheader("Cluster Emergence Timeline")
            
            fig = px.scatter(
                clusters,
                x='start_date',
                y='size',
                size='total_engagement',
                color='dominant_source',
                hover_data=['cluster_id', 'duration_days', 'unique_authors'],
                title="Clusters Over Time (bubble size = engagement)",
                labels={'start_date': 'Start Date', 'size': 'Cluster Size'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Top clusters table
            st.subheader("🏆 Top 20 Largest Clusters")
            
            top_clusters = clusters.nlargest(20, 'size')[[
                'cluster_id', 'size', 'duration_days', 'total_engagement',
                'unique_authors', 'dominant_source'
            ]].copy()
            
            top_clusters.columns = ['Cluster ID', 'Size', 'Duration (days)', 
                                   'Total Engagement', 'Unique Authors', 'Platform']
            
            st.dataframe(top_clusters, hide_index=True, use_container_width=True)
            
            # Individual cluster explorer
            st.subheader("🔍 Explore Individual Cluster")
            
            selected_cluster = st.selectbox(
                "Select Cluster",
                options=clusters['cluster_id'].tolist(),
                index=0
            )
            
            if selected_cluster:
                cluster_posts = df_filtered[df_filtered['cluster_global'] == selected_cluster]
                cluster_info = clusters[clusters['cluster_id'] == selected_cluster].iloc[0]
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Posts", f"{len(cluster_posts):,}")
                    st.metric("Unique Authors", cluster_info['unique_authors'])
                
                with col2:
                    st.metric("Duration", f"{cluster_info['duration_days']} days")
                    st.metric("Total Engagement", f"{cluster_info['total_engagement']:,}")
                
                with col3:
                    st.metric("Dominant Platform", cluster_info['dominant_source'])
                    st.metric("Avg Engagement/Post", f"{cluster_info['avg_engagement']:.1f}")
                
                # Sample posts from cluster
                st.subheader("Sample Posts")
                sample_posts = cluster_posts.nlargest(5, 'interactions')[['timestamp', 'author', 'text', 'interactions', 'source']]
                st.dataframe(sample_posts, hide_index=True, use_container_width=True)
    
    # ==============================================================================
    # TAB 5: DISINFORMATION ANALYSIS
    # ==============================================================================
    
    with tab5:
        st.header("⚠️ Disinformation Markers Analysis")
        
        # Check for disinformation marker columns
        marker_cols = [c for c in df_filtered.columns if c.startswith('marker_')]
        
        if not marker_cols:
            st.warning("⚠️ Disinformation marker data not available")
        else:
            # Marker prevalence
            st.subheader("Disinformation Marker Prevalence")
            
            marker_stats = []
            for col in marker_cols:
                marker_name = col.replace('marker_', '').replace('_', ' ').title()
                count = df_filtered[col].sum()
                percentage = (count / len(df_filtered)) * 100
                marker_stats.append({
                    'Marker Type': marker_name,
                    'Count': int(count),
                    'Percentage': f"{percentage:.2f}%"
                })
            
            marker_df = pd.DataFrame(marker_stats).sort_values('Count', ascending=False)
            
            fig = px.bar(
                marker_df,
                x='Count',
                y='Marker Type',
                orientation='h',
                title="Disinformation Marker Prevalence",
                color='Count',
                color_continuous_scale='Reds'
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(marker_df, hide_index=True, use_container_width=True)
            
            # Marker co-occurrence
            st.subheader("Marker Co-occurrence Analysis")
            
            if 'disinfo_marker_count' in df_filtered.columns:
                posts_with_markers = df_filtered[df_filtered['disinfo_marker_count'] > 0]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Posts with Markers", f"{len(posts_with_markers):,}")
                    st.metric("Percentage", f"{(len(posts_with_markers)/len(df_filtered)*100):.1f}%")
                
                with col2:
                    st.metric("Avg Markers per Post", f"{df_filtered['disinfo_marker_count'].mean():.2f}")
                    st.metric("Max Markers in Single Post", f"{df_filtered['disinfo_marker_count'].max()}")
                
                # Distribution of marker counts
                marker_count_dist = df_filtered['disinfo_marker_count'].value_counts().sort_index()
                
                fig = px.bar(
                    x=marker_count_dist.index,
                    y=marker_count_dist.values,
                    labels={'x': 'Number of Markers', 'y': 'Number of Posts'},
                    title="Distribution of Marker Counts per Post"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Temporal evolution of markers
            st.subheader("Temporal Evolution of Disinformation Markers")
            
            temporal_markers = df_filtered.set_index('timestamp')[marker_cols].resample('W').sum()
            
            # Select top 5 markers for visualization
            top_markers = marker_df.head(5)['Marker Type'].apply(
                lambda x: 'marker_' + x.lower().replace(' ', '_')
            ).tolist()
            
            fig = go.Figure()
            for marker in top_markers:
                if marker in temporal_markers.columns:
                    marker_name = marker.replace('marker_', '').replace('_', ' ').title()
                    fig.add_trace(go.Scatter(
                        x=temporal_markers.index,
                        y=temporal_markers[marker],
                        mode='lines',
                        name=marker_name
                    ))
            
            fig.update_layout(
                title="Top 5 Disinformation Markers Over Time (Weekly)",
                xaxis_title="Date",
                yaxis_title="Occurrences",
                hovermode='x unified',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # ==============================================================================
    # TAB 6: SEARCH & INVESTIGATE
    # ==============================================================================
    
    with tab6:
        st.header("🔍 Search & Investigate")
        
        st.subheader("Search Posts")
        
        search_query = st.text_input("Enter search terms", placeholder="vaccine injury, climate hoax, etc.")
        
        if search_query:
            # Case-insensitive search in text
            search_results = df_filtered[
                df_filtered['text'].str.contains(search_query, case=False, na=False)
            ]
            
            st.metric("Results Found", f"{len(search_results):,}")
            
            if len(search_results) > 0:
                # Show results
                st.subheader("Search Results")
                
                # Sort by engagement
                top_results = search_results.nlargest(20, 'interactions')[
                    ['timestamp', 'author', 'source', 'text', 'interactions']
                ].copy()
                
                top_results['timestamp'] = top_results['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
                top_results['text'] = top_results['text'].str[:200] + '...'
                
                st.dataframe(top_results, hide_index=True, use_container_width=True)
                
                # Export results
                csv = search_results.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name=f"search_results_{search_query.replace(' ', '_')}.csv",
                    mime="text/csv"
                )
        
        st.markdown("---")
        
        # Author lookup
        st.subheader("Author Profile")
        
        author_search = st.text_input("Enter author name", placeholder="username")
        
        if author_search:
            author_posts = df_filtered[df_filtered['author'] == author_search]
            
            if len(author_posts) > 0:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Posts", f"{len(author_posts):,}")
                
                with col2:
                    st.metric("Platforms Used", author_posts['source'].nunique())
                
                with col3:
                    st.metric("Total Engagement", f"{author_posts['interactions'].sum():,}")
                
                with col4:
                    st.metric("Avg Engagement/Post", f"{author_posts['interactions'].mean():.1f}")
                
                # Activity timeline
                author_timeline = author_posts.set_index('timestamp').resample('D').size()
                
                fig = px.line(
                    x=author_timeline.index,
                    y=author_timeline.values,
                    title=f"Activity Timeline: {author_search}",
                    labels={'x': 'Date', 'y': 'Posts'}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Platform distribution
                platform_dist = author_posts['source'].value_counts()
                
                fig = px.pie(
                    values=platform_dist.values,
                    names=platform_dist.index,
                    title="Platform Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Recent posts
                st.subheader("Recent Posts")
                recent = author_posts.nlargest(10, 'timestamp')[
                    ['timestamp', 'source', 'text', 'interactions']
                ].copy()
                recent['timestamp'] = recent['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
                recent['text'] = recent['text'].str[:200] + '...'
                
                st.dataframe(recent, hide_index=True, use_container_width=True)
            else:
                st.info(f"No posts found for author: {author_search}")

# ==============================================================================
# RUN APP
# ==============================================================================

if __name__ == "__main__":
    main()