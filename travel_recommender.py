import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

# Custom CSS for frontend
def add_custom_css():
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #1e1e2f;
            color: white;
            font-family: 'Arial', sans-serif;
        }
        .sidebar .sidebar-content {
            background-color: #29293d;
            color: white;
        }
        .css-1aumxhk, .css-2trqyj, .css-1d391kg {
            color: white !important;
        }
        .stButton>button {
            background-color: #ff4b4b;
            color: white;
            border-radius: 10px;
        }
        .recommendation-card {
            background-color: #29293d;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 15px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Load and preprocess data
@st.cache_data
def load_data():
    try:
        # Load dataset with exact column names
        df = pd.read_csv('indian_tourist_spots.csv')
        
        # Clean column names (remove extra spaces, make lowercase)
        df.columns = df.columns.str.strip().str.lower()
        
        # Rename columns to match code expectations
        column_mapping = {
            'time needed to visit in hrs': 'time_needed_to_visit_hrs',
            'google review rating': 'google_review_rating',
            'entrance fee in inr': 'entrance_fee_inr',
            'airport with 50km radius': 'airport_within_50km',
            'weekly off': 'weekly_off',
            'dslr allowed': 'dslr_allowed',
            'number of google review in lakhs': 'google_reviews_lakhs',
            'best time to visit': 'best_time_to_visit'
        }
        df = df.rename(columns=column_mapping)
        
        # Fill missing values
        num_cols = ['google_review_rating', 'entrance_fee_inr', 
                   'time_needed_to_visit_hrs', 'google_reviews_lakhs']
        for col in num_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        
        # Create enhanced features
        df['combined_features'] = df.apply(
            lambda row: f"{row.get('type', '')} {row.get('significance', '')} "
                       f"{row.get('zone', '')} {row.get('best_time_to_visit', '')}",
            axis=1
        )
        
        # Feature engineering for ML
        df['popularity_score'] = (
            df['google_review_rating'] * np.log1p(df['google_reviews_lakhs'])
        )
        
        return df
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()

# Train ML models
@st.cache_resource
def train_models(df):
    try:
        # TF-IDF for content-based filtering
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(df['combined_features'])
        
        # K-Means clustering
        kmeans = KMeans(n_clusters=10, random_state=42)
        cluster_features = df[['google_review_rating', 'entrance_fee_inr', 
                             'time_needed_to_visit_hrs', 'popularity_score']]
        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform(cluster_features)
        df['cluster'] = kmeans.fit_predict(scaled_features)
        
        return {
            'tfidf': tfidf,
            'tfidf_matrix': tfidf_matrix,
            'kmeans': kmeans,
            'scaler': scaler,
            'df': df
        }
    
    except Exception as e:
        st.error(f"Error training models: {str(e)}")
        st.stop()

# Get user preferences
def get_user_preferences():
    st.sidebar.header("Travel Preferences")
    
    zone = st.sidebar.selectbox("Preferred Zone", ["Any", "Northern", "Southern", "Eastern", "Western", "Central"])
    state = st.sidebar.text_input("Preferred State (leave blank for any)")
    city = st.sidebar.text_input("Preferred City (leave blank for any)")
    place_type = st.sidebar.multiselect("Type of Place", ["Temple", "War Memorial", "Natural Park", "Beach", "Museum", "Historical Site", "Hill Station"])
    significance = st.sidebar.multiselect("Significance", ["Historical", "Religious", "Environmental", "Cultural", "Adventure"])
    time_available = st.sidebar.slider("Time Available (hours)", 1, 12, 4)
    rating_threshold = st.sidebar.slider("Minimum Google Rating", 1.0, 5.0, 3.5, step=0.1)
    airport_needed = st.sidebar.checkbox("Airport within 50km required")
    dslr_allowed = st.sidebar.checkbox("DSLR Photography Allowed")
    
    if st.sidebar.button("Generate Recommendations"):
        return {
            "zone": zone,
            "state": state,
            "city": city,
            "type": place_type,
            "significance": significance,
            "time_available": time_available,
            "rating_threshold": rating_threshold,
            "airport_needed": airport_needed,
            "dslr_allowed": dslr_allowed
        }
    return None

# Display recommendations
def display_recommendations(recommendations):
    st.subheader("✨ AI-Powered Recommendations")
    st.write(f"Showing {len(recommendations)} best matches based on your preferences:")
    
    for idx, row in recommendations.iterrows():
        with st.container():
            st.markdown(f"""
            <div class="recommendation-card">
                <h3>{row['name']}</h3>
                <p><strong>Location:</strong> {row['city']}, {row['state']} ({row['zone']} India)</p>
                <p><strong>Type:</strong> {row['type']} | <strong>Significance:</strong> {row['significance']}</p>
                <p><strong>Rating:</strong> {row['google_review_rating']:.1f} ⭐ | <strong>Reviews:</strong> {row['google_reviews_lakhs']:.1f} lakhs</p>
                <p><strong>Entrance Fee:</strong> ₹{row['entrance_fee_inr']:.0f} | <strong>Visit Time:</strong> {row['time_needed_to_visit_hrs']} hrs</p>
                <p><strong>Best Time:</strong> {row['best_time_to_visit']}</p>
                <p><strong>DSLR:</strong> {row['dslr_allowed']} | <strong>Airport Nearby:</strong> {row['airport_within_50km']}</p>
            </div>
            """, unsafe_allow_html=True)

# Main function
def main():
    add_custom_css()
    st.title("AI Travel Planner - Smart Recommendations")
    st.write("Discover perfect tourist spots using advanced machine learning algorithms.")
    
    # Load data and train models
    df = load_data()
    models = train_models(df)
    
    # Show dataset info
    if st.checkbox("Show dataset summary"):
        st.write(f"Total tourist spots: {len(df)}")
        st.write("Sample data:")
        st.dataframe(df[['name', 'city', 'state', 'type', 'google_review_rating']].head())
    
    # Get user preferences
    preferences = get_user_preferences()
    
    if preferences:
        with st.spinner("Analyzing preferences and generating recommendations..."):
            # Basic filtering first
            filtered_df = models['df'].copy()
            
            if preferences['zone'] != "Any":
                filtered_df = filtered_df[filtered_df['zone'].str.contains(preferences['zone'], case=False)]
            
            if preferences['state']:
                filtered_df = filtered_df[filtered_df['state'].str.contains(preferences['state'], case=False)]
            
            if preferences['city']:
                filtered_df = filtered_df[filtered_df['city'].str.contains(preferences['city'], case=False)]
            
            if preferences['type']:
                filtered_df = filtered_df[filtered_df['type'].isin(preferences['type'])]
            
            if preferences['significance']:
                filtered_df = filtered_df[filtered_df['significance'].isin(preferences['significance'])]
            
            filtered_df = filtered_df[filtered_df['time_needed_to_visit_hrs'] <= preferences['time_available']]
            filtered_df = filtered_df[filtered_df['google_review_rating'] >= preferences['rating_threshold']]
            
            if preferences['airport_needed']:
                filtered_df = filtered_df[filtered_df['airport_within_50km'].str.contains('Yes', case=False, na=False)]
            
            if preferences['dslr_allowed']:
                filtered_df = filtered_df[filtered_df['dslr_allowed'].str.contains('Yes', case=False, na=False)]
            
            # If we have filtered results, apply ML ranking
            if len(filtered_df) > 0:
                # Content-based similarity
                query = " ".join([
                    " ".join(preferences['type']),
                    " ".join(preferences['significance']),
                    preferences['zone']
                ])
                query_vec = models['tfidf'].transform([query])
                cosine_sim = cosine_similarity(query_vec, models['tfidf_matrix'][filtered_df.index]).flatten()
                filtered_df['content_score'] = cosine_sim
                
                # Sort by combined score
                filtered_df['final_score'] = 0.7*filtered_df['content_score'] + 0.3*filtered_df['popularity_score']
                recommendations = filtered_df.sort_values('final_score', ascending=False).head(10)
                
                display_recommendations(recommendations)
            else:
                # If no results with strict filters, show similar recommendations with relaxed filters
                # st.warning("No exact matches found. Showing similar recommendations:")
                similar_df = models['df'].copy()
                
                # Apply only the most important filters
                if preferences['type']:
                    similar_df = similar_df[similar_df['type'].isin(preferences['type'])]
                
                if preferences['significance']:
                    similar_df = similar_df[similar_df['significance'].isin(preferences['significance'])]
                
                similar_df = similar_df[similar_df['google_review_rating'] >= preferences['rating_threshold']]
                
                if len(similar_df) > 0:
                    query = " ".join([
                        " ".join(preferences['type']),
                        " ".join(preferences['significance']),
                        preferences['zone']
                    ])
                    query_vec = models['tfidf'].transform([query])
                    cosine_sim = cosine_similarity(query_vec, models['tfidf_matrix'][similar_df.index]).flatten()
                    similar_df['content_score'] = cosine_sim
                    
                    similar_df['final_score'] = 0.7*similar_df['content_score'] + 0.3*similar_df['popularity_score']
                    recommendations = similar_df.sort_values('final_score', ascending=False).head(10)
                    
                    display_recommendations(recommendations)
                else:
                    st.error("Couldn't find any similar recommendations. Please try different filters.")

if __name__ == "__main__":
    main()