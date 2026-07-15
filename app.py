# app.py
import streamlit as st
import pickle
import pandas as pd

# Page setup
st.set_page_config(page_title="Nepali News Classifier", page_icon="📰", layout="centered")

st.title("📰 Nepali News Category Classifier")
st.write("Enter a Nepali news headline or paragraph, and the AI will classify it into Politics, Sports, Entertainment, or Technology.")
st.markdown("---")

# Load the model
@st.cache_resource
def load_news_model():
    with open("nepali_news_model.pkl", "rb") as f:
        return pickle.load(f)

try:
    artifact = load_news_model()
    pipeline = artifact["pipeline"]
    category_map = artifact["categories"]
except FileNotFoundError:
    st.error("⚠️ 'nepali_news_model.pkl' not found! Please run 'python train_news.py' first to train the model.")
    st.stop()

# User Input Box
user_text = st.text_area("Paste Nepali News Text Here:", height=150, placeholder="उदाहरन: नेपालले अन्तर्राष्ट्रिय खेलमा स्वर्ण पदक जित्न सफल भयो...")

if st.button("Classify Category 🔮", use_container_width=True):
    if user_text.strip() == "":
        st.warning("Please enter some text first.")
    else:
        # 1. Predict Category
        predicted_class_id = pipeline.predict([user_text])[0]
        predicted_category = category_map[predicted_class_id]
        
        # 2. Get Confidence Probabilities
        probabilities = pipeline.predict_proba([user_text])[0]
        
        # Display Best Category with styled boxes
        st.markdown(f"""
            <div style="background-color: #F0FDF4; border-left: 5px solid #22C55E; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
                <h4 style="color: #15803D; margin: 0;">Predicted Category:</h4>
                <p style="font-size: 22px; font-weight: bold; margin: 5px 0 0 0; color: #166534;">{predicted_category}</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Display Probability Breakdown using Pandas & Streamlit Progress Bars
        st.subheader("📊 Confidence Score Breakdown")
        
        prob_df = pd.DataFrame({
            "Category": list(category_map.values()),
            "Confidence": probabilities
        })
        
        for idx, row in prob_df.iterrows():
            col1, col2 = st.columns([1, 3])
            with col1:
                st.write(f"**{row['Category'].split()[-1]}**") # Displays English tag
            with col2:
                st.progress(float(row['Confidence']))
                st.caption(f"{row['Confidence'] * 100:.1f}% confidence")
