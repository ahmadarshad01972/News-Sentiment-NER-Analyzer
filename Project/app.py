import streamlit as st
import requests
import json
import csv
from bs4 import BeautifulSoup
from textblob import TextBlob
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from deep_translator import GoogleTranslator
from wordcloud import WordCloud, STOPWORDS
import spacy
import subprocess
import importlib.util
import pandas as pd
import plotly.express as px
import datetime
from io import StringIO

# Install spaCy model if not found (works on Streamlit Cloud)
model_name = "en_core_web_sm"
try:
    nlp = spacy.load(model_name)
except:
    subprocess.run(["python", "-m", "spacy", "download", model_name])
    importlib.util.invalidate_caches()
    nlp = spacy.load(model_name)

st.set_page_config(page_title="News Sentiment Analysis", layout="wide")
st.title("📰 News Sentiment & NER Analyzer")

# Sidebar
with st.sidebar:
    query = st.text_input("Search Bing News for:", value="Technology")
    target_lang = st.selectbox("Translate to language:", ['en', 'es', 'fr', 'ur'])
    filter_sentiment = st.checkbox("Filter by sentiment?")
    sentiment_type = st.selectbox("Choose sentiment type:", ["Positive", "Neutral", "Negative"]) if filter_sentiment else None
    show_ner = st.checkbox("Show Named Entities", value=True)
    theme = st.radio("Choose Theme", ["Light", "Dark"])
    st.caption(f"Last updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    with st.expander("ℹ️ About This App"):
        st.write("This tool fetches Bing News, analyzes sentiment and named entities, and displays results in charts and tables.")
    analyze_button = st.button("Analyze")

if theme == "Dark":
    st.markdown("""
        <style>
        body { background-color: #111 !important; color: #eee !important; }
        </style>
        """, unsafe_allow_html=True)

# Main logic
if analyze_button:
    url = f"https://www.bing.com/news/search?q={query}"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'lxml')
    headlines = [a.get_text() for a in soup.find_all("a", class_="title")]

    selected = []
    json_results = []
    word_freq = defaultdict(int)
    noun_phrases = []
    sentiments = {"Positive": 0, "Neutral": 0, "Negative": 0}

    for text in headlines:
        blob = TextBlob(text)
        sentiment = "Neutral"
        pol = blob.sentiment.polarity
        sub = blob.sentiment.subjectivity

        if pol > 0.1:
            sentiment = "Positive"
        elif pol < -0.1:
            sentiment = "Negative"

        if not filter_sentiment or sentiment == sentiment_type:
            translated = GoogleTranslator(source='auto', target=target_lang).translate(text)
            sentiments[sentiment] += 1
            json_results.append({
                "Headline": text,
                "Translated": translated,
                "Sentiment": sentiment,
                "Polarity": pol,
                "Subjectivity": sub
            })
            selected.append(text)

            for word in text.lower().split():
                if word not in STOPWORDS and len(word) > 1:
                    word_freq[word] += 1
            noun_phrases.extend(blob.noun_phrases)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Sentiment", "☁️ Word Cloud", "🔑 Keywords", "🧠 Named Entities", "📋 Data Table"])

    with tab1:
        st.subheader("Sentiment Summary")
        df_sent = pd.DataFrame.from_dict(sentiments, orient='index', columns=['Count']).reset_index()
        df_sent.columns = ['Sentiment', 'Count']
        fig = px.pie(df_sent, names='Sentiment', values='Count', title='Sentiment Distribution')
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Word Cloud")
        cloud = WordCloud(width=800, height=400, background_color='black', stopwords=STOPWORDS).generate(" ".join(selected))
        st.image(cloud.to_array())

    with tab3:
        st.subheader("Top Keywords")
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        df_keywords = pd.DataFrame(top_words, columns=['Keyword', 'Count'])
        fig_keywords = px.bar(df_keywords, x='Keyword', y='Count', title="Top 10 Keywords", color='Count', color_continuous_scale='viridis')
        st.plotly_chart(fig_keywords, use_container_width=True)
        st.write(df_keywords)

    if show_ner:
        with tab4:
            st.subheader("Named Entities")
            for text in selected:
                doc = nlp(text)
                ents = [(ent.text, ent.label_) for ent in doc.ents]
                if ents:
                    with st.expander(f"{text}"):
                        for ent, label in ents:
                            st.markdown(f"<span style='color:green'>• {ent}</span> (<code>{label}</code>)", unsafe_allow_html=True)

    with tab5:
        st.subheader("All Analyzed Headlines")
        df_all = pd.DataFrame(json_results)
        st.dataframe(df_all)

    csv_buffer = StringIO()
    df_all.to_csv(csv_buffer, index=False)
    st.download_button("Download CSV", csv_buffer.getvalue(), file_name="headlines.csv", mime="text/csv")

    st.download_button("Download JSON", json.dumps(json_results, indent=2), file_name="headlines.json", mime="application/json")
