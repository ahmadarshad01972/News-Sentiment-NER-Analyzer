import streamlit as st
import requests
import json
from bs4 import BeautifulSoup
from textblob import TextBlob
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from deep_translator import GoogleTranslator
from wordcloud import WordCloud, STOPWORDS
import pandas as pd
import plotly.express as px
import datetime
import io
# Load spaCy model


# App setup
st.set_page_config(page_title="News Sentiment Analysis", layout="wide")
st.title("📰 News Sentiment & NER Analyzer")

# Sidebar
with st.sidebar:
    query = st.text_input("🔍 Search Bing News for:", value="Technology")
    target_lang = st.selectbox("🌐 Translate to:", ['en', 'es', 'fr', 'ur'])
    filter_sentiment = st.checkbox("🎯 Filter by sentiment?")
    sentiment_type = st.selectbox("Choose sentiment type:", ["Positive", "Neutral", "Negative"]) if filter_sentiment else None
    show_ner = st.checkbox("🧠 Show Named Entities", value=True)
    theme = st.radio("🌓 Theme", ["Light", "Dark"])
    st.caption(f"🕒 Last updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    analyze_button = st.button("📊 Analyze")

# Apply dark theme styling
if theme == "Dark":
    st.markdown("""
        <style>
        body, .stApp { background-color: #111 !important; color: #eee !important; }
        </style>
        """, unsafe_allow_html=True)

# Start analysis
if analyze_button:
    url = f"https://www.bing.com/news/search?q={query}"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'lxml')
    headlines = [a.get_text() for a in soup.find_all("a", class_="title")]

    results = []
    selected_texts = []
    word_freq = defaultdict(int)
    noun_phrases = []
    sentiments = {"Positive": 0, "Neutral": 0, "Negative": 0}

    for text in headlines:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity

        sentiment = "Neutral"
        if polarity > 0.1:
            sentiment = "Positive"
        elif polarity < -0.1:
            sentiment = "Negative"

        if not filter_sentiment or sentiment == sentiment_type:
            translated = GoogleTranslator(source='auto', target=target_lang).translate(text)
            sentiments[sentiment] += 1
            selected_texts.append(text)
            results.append({
                "Headline": text,
                "Translated": translated,
                "Sentiment": sentiment,
                "Polarity": polarity,
                "Subjectivity": subjectivity
            })

            for word in text.lower().split():
                if word not in STOPWORDS and len(word) > 1:
                    word_freq[word] += 1
            #noun_phrases.extend(blob.noun_phrases)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📈 Sentiment", "☁️ Word Cloud", "🔑 Keywords", "🧠 Named Entities", "📋 Data Table", "ℹ️ About"])

    # Tab 1: Sentiment chart
    with tab1:
        st.subheader("Sentiment Distribution")
        df_sent = pd.DataFrame(sentiments.items(), columns=["Sentiment", "Count"])
        fig = px.pie(df_sent, names='Sentiment', values='Count', title='Sentiment Pie Chart')
        st.plotly_chart(fig, use_container_width=True)

    # Tab 2: Word Cloud
    # Tab 2: Word Cloud
    with tab2:
        st.subheader("Word Cloud")
        if selected_texts:
            wc = WordCloud(width=800, height=400, background_color='black', stopwords=STOPWORDS).generate(" ".join(selected_texts))
            st.image(wc.to_array())
        else:
            st.warning("No headlines available to generate a word cloud.")


    # Tab 3: Keywords
    with tab3:
        st.subheader("Top Keywords")
        top_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        df_kw = pd.DataFrame(top_keywords, columns=["Keyword", "Count"])
        fig_kw = px.bar(df_kw, x="Keyword", y="Count", color="Count", title="Top 10 Keywords")
        st.plotly_chart(fig_kw, use_container_width=True)
        st.write(df_kw)

    
        # Tab 4: Named Entity Recognition
    if show_ner:
        with tab4:
            st.subheader("Named Entities")
            st.header(" 🚧Under Construction 🚧")
            
            st.markdown("""
<div style='background-color:#2c2f33; color:#f1f1f1; padding:15px; border-left:5px solid #f39c12; border-radius:10px; font-size:16px;'>
    <strong>Note:</strong> We initially planned to use <code style="background-color:#1e1e1e; color:#2ecc71;">spaCy</code> for broader NLP tasks, 
    but due to limitations on platforms like Streamlit Cloud where some models can't be loaded easily, 
    we use <code style="background-color:#1e1e1e; color:#2ecc71;">spaCy</code> only for Named Entity Recognition (NER). 
    This ensures that we retain key features without affecting deployment.
</div>
""", unsafe_allow_html=True)




    # Tab 5: Table and Download
        # Tab 5: Table and Download
    with tab5:
        st.subheader("Analyzed Headlines")
        df = pd.DataFrame(results)
        st.dataframe(df)

        # CSV Download Button
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        st.download_button("⬇️ Download CSV", csv_buffer.getvalue(), file_name="headlines.csv", mime="text/csv")

        # JSON download
        st.download_button("⬇️ Download JSON", json.dumps(results, indent=2), file_name="headlines.json", mime="application/json")

    # Tab 6: About
    with tab6:
        st.markdown("<h2 style='color:#4CAF50;'>About This App</h2>", unsafe_allow_html=True)
        st.markdown("""
        This app performs **sentiment analysis** and **named entity recognition (NER)** on Bing News headlines.

        #### 🔍 Features:
        - Live scraping of news using Bing
        - Sentiment detection with visualizations
        - Word Cloud generation
        - Entity highlighting using spaCy
        - Multi-language translation

        ---

        #### 👨‍💻 Developer Info:
        <a href="https://github.com/ahmadarshad01972" target="_blank">
            <img src="https://img.shields.io/badge/GitHub-ahmadarshad01972-181717?style=for-the-badge&logo=github" height="30">
        </a>
        &nbsp;
        <a href="https://www.linkedin.com/in/ahmad-arshad-04b473322/" target="_blank">
            <img src="https://img.shields.io/badge/LinkedIn-Ahmad%20Arshad-0077B5?style=for-the-badge&logo=linkedin" height="30">
        </a>
        """, unsafe_allow_html=True)
