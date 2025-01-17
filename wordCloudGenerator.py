import streamlit as st
from wordcloud import WordCloud, STOPWORDS
from PyPDF2 import PdfReader
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import requests
from io import BytesIO
import os
import datetime

# File path to store global word cloud count
COUNT_FILE = "wordcloud_count.txt"

def load_wordcloud_count():
    """Load the global word cloud count from a file."""
    if os.path.exists(COUNT_FILE):
        with open(COUNT_FILE, "r") as file:
            try:
                return int(file.read().strip())
            except ValueError:
                return 0
    return 0

def save_wordcloud_count(count):
    """Save the updated global word cloud count to a file."""
    with open(COUNT_FILE, "w") as file:
        file.write(str(count))

# Initialize global word cloud count
global_wordcloud_count = load_wordcloud_count()

# Set up session state to keep track of wordclouds if not already done
if 'wordcloud_history' not in st.session_state:
    st.session_state['wordcloud_history'] = deque(maxlen=10)

# Title
st.title("Ankit's WordCloud App")
st.header("App that takes PDF file as input, extracts text, preprocesses it, removes stop words, and builds a word cloud")

# Sidebar for user inputs
st.sidebar.header("WordCloud Configuration")
colormap_options = sorted(plt.colormaps())
colormap = st.sidebar.selectbox("Select colormap", colormap_options, index=colormap_options.index("viridis"))
max_words = st.sidebar.slider("Select max words", min_value=400, max_value=800, value=500)
background_color = st.sidebar.color_picker("Select background color", "#ffffff")
secret_text = st.sidebar.text_input("Optional Secret Text", "")

# Company logos for customization
company_logos = {
    "Default": None,
    "Google": "https://upload.wikimedia.org/wikipedia/commons/4/4a/Logo_2013_Google.png",
    "Microsoft": "https://upload.wikimedia.org/wikipedia/commons/4/44/Microsoft_logo.svg",
    "Apple": "https://upload.wikimedia.org/wikipedia/commons/f/fa/Apple_logo_black.svg",
    "Amazon": "https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo_black.svg"
}

logo_selection = st.sidebar.selectbox("Customize with logo mask", list(company_logos.keys()))
selected_logo_url = company_logos[logo_selection]

# Upload PDF file
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

def extract_text_from_pdf(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

def get_mask_from_logo(logo_url):
    """Generate a mask from a logo image URL."""
    if logo_url:
        response = requests.get(logo_url)
        logo_image = Image.open(BytesIO(response.content)).convert("L")
        return np.array(logo_image)
    return None

if uploaded_file is not None:
    processing_message = st.empty()
    processing_message.subheader("Processing...")
    pdf_text = extract_text_from_pdf(uploaded_file)

    if pdf_text:
        processing_message.empty()
        words = pdf_text.split()
        stopwords = set(STOPWORDS)
        filtered_words = [word for word in words if word.lower() not in stopwords]
        processed_text = " ".join(filtered_words)

        mask = get_mask_from_logo(selected_logo_url)
        wordcloud = WordCloud(
            width=800, height=400, max_words=max_words, colormap=colormap,
            background_color=background_color, stopwords=stopwords, mask=mask
        ).generate(processed_text)

        wordcloud_image = wordcloud.to_image()

        if secret_text != "Ankit@Sharma":
            wordcloud_image = add_watermark(wordcloud_image, "Generated @ Ankit's WordCloud")

        st.image(wordcloud_image)

        global_wordcloud_count += 1
        save_wordcloud_count(global_wordcloud_count)
        st.write(f"WordClouds Generated Globally: {global_wordcloud_count}")

        creation_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.wordcloud_history.append((colormap, wordcloud_image, creation_time))

        st.subheader("WordCloud History")
        for cmap, wc_img, timestamp in st.session_state.wordcloud_history:
            st.write(f"Colormap: {cmap}, Created on: {timestamp}")
            st.image(wc_img)
    else:
        processing_message.empty()
        st.error("Unable to extract text from the uploaded PDF.")
else:
    st.info("Please upload a PDF file to proceed.")
