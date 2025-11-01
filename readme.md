# Disinformation Ecosystem Dashboard

This Streamlit application visualizes a multi-platform disinformation ecosystem, analyzing narratives, key influencers, and network structures.

## Setup Instructions

1.  **Clone or download this repository.**

2.  **Install Dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Place Data Files:**
    Ensure the following files (generated from the Colab pipeline) are placed in the correct directories:
    - `data/processed_data_with_features.pkl`
    - `data/final_clusters_with_metadata.pkl`
    - `results/networks/author_network.gexf`

4.  **Run the Application:**
    Navigate to the root directory of the project in your terminal and run:
    ```bash
    streamlit run dashboard_app.py
    ```

The application will open in your default web browser.