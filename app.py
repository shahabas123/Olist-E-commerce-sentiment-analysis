import streamlit as st
import pickle
import pandas as pd
import plotly.express as px
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import requests
import os
from io import BytesIO

def main():
    def set_background_image(image_url):
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("{image_url}");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }}
            </style>
            """,
            unsafe_allow_html=True)
    set_background_image('https://i.pinimg.com/736x/48/7e/13/487e133428eedbbafed3191d765db61d.jpg')
    st.markdown(
        """
        <style>
        /* Set all text to black */
        * {
            color: #f58799 !important;
        }
        /* Set dropdown text to white */
        [data-baseweb="select"] .css-1wa3eu0 {
        color: white !important;
        }

        /* Set dropdown options text to their default color */
        [data-baseweb="select"] {
            color: #333 !important;
            
        }

        /* Set the predict button text to its default color */
        button {
            color: white !important;
            background-color: #ad9884 !important; /* Blue button */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.sidebar.title("Select Page:")
    page = st.sidebar.radio("Go to", ["Home", "Analysis", "Prediction" , "Contact"])

    st.markdown("""
                <style>
                div[data-baseweb="select"] {
            background-color: #000033;

            border-radius: 2px;
            padding: 2px;
        }

                 # @keyframes zoomEffect {
                 #        0% { background-size: 150%; }
                 #        100% { background-size: 140%; }
                 #    }
                    [data-testid="stSidebar"] {
                        background: url("https://i.pinimg.com/736x/67/93/56/6793562de5e28d84a8bbc16c7029bf00.jpg") no-repeat center center;
                        background-size: cover;
                        margin-top: 56px;
                        margin-bottom:-50px;
                        #padding-bottom: 300px;
                        animation: zoomEffect 0.45s infinite alternate;
                    }
                   </style>
            """, unsafe_allow_html=True)


    if page == "Home":
        st.title(":bar_chart: ****E-commerce Sentiment Analysis****")
        st.markdown("""
        ### **Project Overview**  
        This app analyzes customer reviews from Olist (Brazilian e-commerce) to:  
        - Predict sentiment (Positive/Neutral/Negative) from reviews.  
        - Track trends and identify product issues.
                
        ### **Dataset**  
        - Source: [Olist Public Dataset](https://www.kaggle.com/olistbr/brazilian-ecommerce)  
        - Contains: 100k+ orders, reviews in Portuguese (translated to English).  
        - Features: Review scores (1-5), product categories, timestamps.  
        """)       
        
    elif page=="Analysis":
        def load_and_merge_data():
            reviews = pd.read_csv("olist_order_reviews_dataset.csv")
            items = pd.read_csv("olist_order_items_dataset.csv")
            products = pd.read_csv("olist_products_dataset.csv")
    
    
            merged = pd.merge(items, products, on='product_id', how='left')
            merged = pd.merge(merged, reviews, on='order_id', how='left')
    
    
            merged['review_date'] = pd.to_datetime(merged['review_creation_date'])
            return merged

        def analytics_dashboard():
            st.title("📈 Brazilian E-Commerce Insights Dashboard")
    
            with st.spinner('Loading and processing data...'):
                data = load_and_merge_data()
    
    
            st.subheader("Sentiment Trend Over Time")
            monthly_sentiment = data.resample('M', on='review_date')['review_score'].mean()
            fig1 = px.line(monthly_sentiment, 
                        title="Monthly Average Review Score",
                        labels={'value': 'Average Score'})
            st.plotly_chart(fig1, use_container_width=True)
    
    # Chart 2: Negative Spike Alert
            st.subheader("⚠️ Negative Sentiment Alerts")
            negative_reviews = data[data['review_score'] <= 2]
            daily_negative = negative_reviews.resample('D', on='review_date').size()
    
            if not daily_negative.empty:
                if daily_negative.max() > 2 * daily_negative.mean():
                    st.warning(f"Alert: Negative reviews spiked on {daily_negative.idxmax().date()}!")
                fig2 = px.bar(daily_negative, 
                            title="Daily Negative Reviews (1-2 Stars)",
                            labels={'value': 'Number of Reviews'})
                st.plotly_chart(fig2, use_container_width=True)
    
    # Chart 3: Worst Categories
            st.subheader("🚨 Worst Performing Categories")
            if 'product_category_name' in data.columns:
                category_negative = (
                    data.groupby('product_category_name')
                    .apply(lambda x: (x['review_score'] <= 2).mean())
                    .sort_values(ascending=False)
                    .head(10)
                )
                fig3 = px.bar(category_negative,
                            title="Top 10 Categories by Negative Review Ratio",
                            labels={'value': 'Negative Review Ratio'})
                st.plotly_chart(fig3, use_container_width=True)
            else:
                st.warning("Product category information not available in the dataset")


        analytics_dashboard()




    elif page=='Prediction':
        st.title("📝 Predict Review Sentiment")

        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            st.warning("Downloading NLTK stopwords... (this will only happen once)")
            nltk.download('stopwords')
 
        stemmer = PorterStemmer()  
        stop_words = set(stopwords.words('english'))  

        MODEL_URL = "https://github.com/shahabas123/Olist-E-commerce-sentiment-analysis/releases/download/rf-model-v1.0/rf_model.pkl"
        VECTORIZER_PATH = "tfidf_vectorizer.pkl"

        @st.cache_resource
        def load_model_from_release():
            try:
                with st.spinner("Downloading model from GitHub release..."):
                    import requests
                    import tempfile

                    response = requests.get(MODEL_URL)
                    response.raise_for_status()

                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
                        tmp_file.write(response.content)
                        tmp_path = tmp_file.name

                    with open(tmp_path, 'rb') as f:
                        model = pickle.load(f)
                    return model    
            except Exception as e:
                st.error(f"Failed to download model: {str(e)}")
                return None    
            

        try:
            vec = pickle.load(open(VECTORIZER_PATH, 'rb'))
            rf = load_model_from_release()
            if rf is None:
                st.stop()
        except Exception as e:
            st.error(f"Failed to load files: {str(e)}")
            st.stop()



        # try:
        #     with open('rf_model.pkl', 'rb') as model_file, \
        #          open('tfidf_vectorizer.pkl', 'rb') as vec_file:
        #         rf = pickle.load(model_file)
        #         vec = pickle.load(vec_file)
        # except Exception as e:
        #     st.error("Model loading failed. Please check the model files.")
        #     st.stop()

        user_input = st.text_area("Enter your review in English:", "")

        if st.button("Predict"):
            if not user_input.strip():
                st.warning("Please enter a review")
            else:
            # Simple preprocessing (matches your training)
                text = re.sub('[^a-zA-Z0-9 ]', '', user_input).lower()
                words = [stemmer.stem(word) for word in text.split() if word not in stop_words]
                processed_text = ' '.join(words)
            
            # Predict
                X = vec.transform([processed_text])
                score = rf.predict(X)[0]

                if score >= 4:
                    st.success("Positive Review")
                elif score == 3:
                    st.warning("Neutral Review")
                else:
                    st.error("Negative Review")
    
        



    elif page=='Contact':
        st.title("📞 Contact Page")
        st.write("For Inquiries, you can reach me at:")
        st.write("📧 Email: shahabasali751@gmail.com")
        st.write("GitHub: [GitHub](https://github.com/shahabas123)")
        st.write("LinkedIn: [LinkedIn](https://www.linkedin.com/in/shahabas-ali-8-/)")

    














main()    