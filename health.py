import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Healthcare Recommendation System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
    .recommendation-card {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #2E86AB;
    }
    .doctor-card {
        background-color: #f9f9f9;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .metric-card {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_healthcare_data():
    doctors = pd.read_csv("indian_doctors.csv")
    symptoms_conditions = pd.read_csv("indian_symptoms.csv")
    return doctors, symptoms_conditions

class HealthcareRecommendationSystem:
    def __init__(self, doctors_df, symptoms_df):
        self.doctors_df = doctors_df
        self.symptoms_df = symptoms_df
        self.vectorizer = TfidfVectorizer()
        self.scaler = StandardScaler()

    def recommend_by_symptoms(self, user_symptoms, location_preference=None, max_fee=None):
        matching_specs = []
        for symptom in user_symptoms:
            matches = self.symptoms_df[
                self.symptoms_df['symptom'].str.contains(symptom.lower(), case=False, na=False)
            ]
            if not matches.empty:
                matching_specs.extend(matches['recommended_specialization'].tolist())

        if not matching_specs:
            return pd.DataFrame()

        recommended_doctors = self.doctors_df[
            self.doctors_df['specialization'].isin(matching_specs)
        ].copy()

        if location_preference:
            recommended_doctors = recommended_doctors[
                recommended_doctors['location'].str.contains(location_preference, case=False, na=False)
            ]

        if max_fee:
            recommended_doctors = recommended_doctors[
                recommended_doctors['consultation_fee'] <= max_fee
            ]

        recommended_doctors['recommendation_score'] = (
            recommended_doctors['rating'] * 0.4 +
            (recommended_doctors['experience'] / recommended_doctors['experience'].max()) * 0.3 +
            ((recommended_doctors['consultation_fee'].max() - recommended_doctors['consultation_fee']) /
             recommended_doctors['consultation_fee'].max()) * 0.3
        )

        return recommended_doctors.sort_values('recommendation_score', ascending=False)

    def recommend_similar_doctors(self, doctor_id, n_recommendations=5):
        target_doctor = self.doctors_df[self.doctors_df['doctor_id'] == doctor_id]
        if target_doctor.empty:
            return pd.DataFrame()

        target_spec = target_doctor['specialization'].iloc[0]
        target_rating = target_doctor['rating'].iloc[0]

        similar_doctors = self.doctors_df[
            (self.doctors_df['specialization'] == target_spec) &
            (self.doctors_df['doctor_id'] != doctor_id)
        ].copy()

        if similar_doctors.empty:
            return pd.DataFrame()

        similar_doctors['similarity_score'] = (
            1 - abs(similar_doctors['rating'] - target_rating) / 5.0
        )

        return similar_doctors.sort_values('similarity_score', ascending=False).head(n_recommendations)

def main():
    st.markdown('<h1 class="main-header">🏥 Healthcare Recommendation System</h1>', unsafe_allow_html=True)
    doctors_df, symptoms_df = load_healthcare_data()
    recommender = HealthcareRecommendationSystem(doctors_df, symptoms_df)

    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", [
        "Symptom-Based Recommendations",
        "Doctor Search",
        "Similar Doctors",
        "Healthcare Analytics"
    ])

    if page == "Symptom-Based Recommendations":
        st.header("🔍 Find Doctors Based on Your Symptoms")
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Enter Your Symptoms")
            symptoms_input = st.text_area(
                "Describe your symptoms (separate multiple symptoms with commas):",
                placeholder="e.g., chest pain, headache, fever",
                height=100
            )

            st.subheader("Preferences (Optional)")
            location_pref = st.selectbox(
                "Preferred Location:",
                ["Any"] + sorted(doctors_df['location'].unique().tolist())
            )

            max_fee = st.slider(
                "Maximum Consultation Fee (₹):",
                min_value=500,
                max_value=3000,
                value=2000,
                step=100
            )

            if st.button("Get Recommendations", type="primary"):
                if symptoms_input:
                    symptoms_list = [s.strip() for s in symptoms_input.split(',')]
                    location_filter = None if location_pref == "Any" else location_pref
                    recommendations = recommender.recommend_by_symptoms(symptoms_list, location_filter, max_fee)

                    if not recommendations.empty:
                        st.success(f"Found {len(recommendations)} recommended doctors!")
                        for idx, (_, doctor) in enumerate(recommendations.head(5).iterrows()):
                            with st.container():
                                st.markdown(f"""
                                <div class="recommendation-card">
                                    <h3>#{idx+1} {doctor['name']}</h3>
                                    <p><strong>Specialization:</strong> {doctor['specialization']}</p>
                                    <p><strong>Experience:</strong> {doctor['experience']} years</p>
                                    <p><strong>Rating:</strong> ⭐ {doctor['rating']}/5.0</p>
                                    <p><strong>Location:</strong> {doctor['location']}</p>
                                    <p><strong>Consultation Fee:</strong> ₹{doctor['consultation_fee']}</p>
                                    <p><strong>Recommendation Score:</strong> {doctor['recommendation_score']:.2f}</p>
                                </div>
                                """, unsafe_allow_html=True)
                    else:
                        st.warning("No doctors found matching your criteria. Try adjusting your filters.")
                else:
                    st.error("Please enter your symptoms to get recommendations.")

        with col2:
            st.subheader("Common Symptoms Guide")
            st.dataframe(symptoms_df, use_container_width=True)

    elif page == "Doctor Search":
        st.header("👨‍⚕️ Search and Filter Doctors")
        specialization_filter = st.selectbox("Specialization:", ["All"] + sorted(doctors_df['specialization'].unique()))
        location_filter = st.selectbox("Location:", ["All"] + sorted(doctors_df['location'].unique()))
        min_rating = st.slider("Minimum Rating:", 1.0, 5.0, 4.0, 0.1)
        max_fee_search = st.slider("Maximum Fee (₹):", 500, 3000, 2000, 100)

        filtered_doctors = doctors_df.copy()
        if specialization_filter != "All":
            filtered_doctors = filtered_doctors[filtered_doctors['specialization'] == specialization_filter]
        if location_filter != "All":
            filtered_doctors = filtered_doctors[filtered_doctors['location'] == location_filter]
        filtered_doctors = filtered_doctors[
            (filtered_doctors['rating'] >= min_rating) &
            (filtered_doctors['consultation_fee'] <= max_fee_search)
        ]

        st.subheader(f"Found {len(filtered_doctors)} doctors")
        for _, doctor in filtered_doctors.iterrows():
            with st.container():
                st.markdown(f"""
                <div class="doctor-card">
                    <h4>{doctor['name']}</h4>
                    <p><strong>Specialization:</strong> {doctor['specialization']} | 
                        <strong>Experience:</strong> {doctor['experience']} years | 
                        <strong>Rating:</strong> ⭐ {doctor['rating']}</p>
                    <p><strong>Location:</strong> {doctor['location']} | 
                        <strong>Fee:</strong> ₹{doctor['consultation_fee']}</p>
                </div>
                """, unsafe_allow_html=True)

    elif page == "Similar Doctors":
        st.header("🔄 Find Similar Doctors")
        selected_doctor = st.selectbox("Select a doctor:", options=doctors_df['doctor_id'],
                                         format_func=lambda x: f"{doctors_df[doctors_df['doctor_id']==x]['name'].iloc[0]} - {doctors_df[doctors_df['doctor_id']==x]['specialization'].iloc[0]}")
        if st.button("Find Similar Doctors"):
            similar_doctors = recommender.recommend_similar_doctors(selected_doctor)
            if not similar_doctors.empty:
                st.success(f"Found {len(similar_doctors)} similar doctors!")
                for idx, (_, doctor) in enumerate(similar_doctors.iterrows()):
                    st.markdown(f"""
                    <div class="doctor-card">
                        <h4>#{idx+1} {doctor['name']}</h4>
                        <p><strong>Specialization:</strong> {doctor['specialization']} | 
                            <strong>Experience:</strong> {doctor['experience']} years | 
                            <strong>Rating:</strong> ⭐ {doctor['rating']}</p>
                        <p><strong>Location:</strong> {doctor['location']} | 
                            <strong>Fee:</strong> ₹{doctor['consultation_fee']} | 
                            <strong>Similarity Score:</strong> {doctor['similarity_score']:.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("No similar doctors found.")

    elif page == "Healthcare Analytics":
       st.header("📊 Healthcare Analytics Dashboard")
       col1, col2 = st.columns(2)

       with col1:
           st.subheader("Doctors by Specialization")
           st.bar_chart(doctors_df['specialization'].value_counts())

           st.subheader("Rating Distribution")
           import matplotlib.pyplot as plt
           fig, ax = plt.subplots()
           ax.hist(doctors_df['rating'], bins=10, color='skyblue', edgecolor='black')
           ax.set_title("Rating Distribution")
           ax.set_xlabel("Rating")
           ax.set_ylabel("Count")
           st.pyplot(fig)

           st.subheader("Top 10 Most Experienced Doctors")
           top_exp = doctors_df.sort_values("experience", ascending=False).head(10)[["name", "experience"]]
           st.bar_chart(top_exp.set_index("name"))

       with col2:
           st.subheader("Fees by Specialization")
           fee_by_spec = doctors_df.groupby('specialization')['consultation_fee'].mean().sort_values(ascending=False)
           st.bar_chart(fee_by_spec)

           st.subheader("Experience vs Rating")
           st.scatter_chart(doctors_df[['experience', 'rating']].set_index('experience'))

           st.subheader("Rating by Specialization")
           avg_rating_spec = doctors_df.groupby('specialization')['rating'].mean().sort_values(ascending=False)
           st.bar_chart(avg_rating_spec)

       st.subheader("Summary Statistics")
       summary_stats = doctors_df.groupby('specialization').agg({
           'rating': ['mean', 'count'],
           'consultation_fee': 'mean',
           'experience': 'mean'
       }).round(2)
       summary_stats.columns = ['Avg Rating', 'Doctor Count', 'Avg Fee (₹)', 'Avg Experience (yrs)']
       st.dataframe(summary_stats)

       # Pie chart for location distribution
       st.subheader("Doctor Distribution by Location")
       import plotly.express as px
       loc_counts = doctors_df['location'].value_counts().reset_index()
       loc_counts.columns = ['location', 'count']
       fig = px.pie(loc_counts, names='location', values='count', title='Doctors by Location')
       st.plotly_chart(fig)

if __name__ == "__main__":
    main()
