import streamlit as st
from wordcloud import WordCloud, STOPWORDS
from PyPDF2 import PdfReader
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque
from PIL import ImageDraw, ImageFont, Image

# Set up session state to keep track of wordclouds if not already done
if 'wordcloud_history' not in st.session_state:
    st.session_state['wordcloud_history'] = deque(maxlen=10)  # Keep up to 10 wordclouds

# Title
st.title("Ankit's WordCloud App")
st.header("Create a streamlit app that takes pdf file as an input, extracts text, preprocess text, removes stop words and build a wordcloud of 500 words. Let the user change the colormap parameter from a dropdown with all available colormap options. Remove the axis of the graph. Keep a history of all generated wordclouds in a separate pane for the user to use till session lasts")

# Sidebar for user inputs
st.sidebar.header("WordCloud Configuration")
colormap_options = sorted(plt.colormaps())
colormap = st.sidebar.selectbox("Select colormap", colormap_options, index=colormap_options.index("viridis"))

# Max words slider
max_words = st.sidebar.slider("Select max words", min_value=400, max_value=800, value=500)

# Background color picker
background_color = st.sidebar.color_picker("Select background color", "#ffffff")

# Secret text input
secret_text = st.sidebar.text_input("Optional Secret Text", "")

# Upload PDF file
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

def extract_text_from_pdf(file):
    """Extract text from uploaded PDF file."""
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

def add_watermark(wordcloud_image, text):
    """Add watermark text to a PIL image at the bottom right."""
    watermark_font = ImageFont.load_default()
    image = wordcloud_image.convert("RGBA")
    watermark = Image.new("RGBA", image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(watermark)
    text_width, text_height = draw.textsize(text, font=watermark_font)
    position = (image.width - text_width - 10, image.height - text_height - 10)
    draw.text(position, text, font=watermark_font, fill=(255, 255, 255, 128))
    combined = Image.alpha_composite(image, watermark)
    return combined.convert("RGB")

# Process uploaded PDF
if uploaded_file is not None:
    st.subheader("Processing...")
    pdf_text = extract_text_from_pdf(uploaded_file)

    if pdf_text:
        # Preprocess text (remove special characters and stopwords)
        words = pdf_text.split()
        stopwords = set(STOPWORDS)
        filtered_words = [word for word in words if word.lower() not in stopwords]
        processed_text = " ".join(filtered_words)

        # Generate wordcloud
        wordcloud = WordCloud(width=800, height=400, max_words=max_words, colormap=colormap, background_color=background_color, stopwords=stopwords).generate(processed_text)

        # Convert wordcloud to image for watermarking
        wordcloud_image = wordcloud.to_image()

        # Add watermark if secret_text is incorrect
        if secret_text != "Ankit@Sharma":
            wordcloud_image = add_watermark(wordcloud_image, "Generated @ Ankit's WordCloud")

        # Display wordcloud
        st.image(wordcloud_image)

        # Store wordcloud history
        st.session_state.wordcloud_history.append((colormap, wordcloud_image))

        # Display history pane
        st.subheader("WordCloud History")
        for cmap, wc_img in st.session_state.wordcloud_history:
            st.write(f"Colormap: {cmap}")
            st.image(wc_img)
    else:
        st.error("Unable to extract text from the uploaded PDF.")
else:
    st.info("Please upload a PDF file to proceed.")
