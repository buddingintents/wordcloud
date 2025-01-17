import streamlit as st
from wordcloud import WordCloud, STOPWORDS
from PyPDF2 import PdfReader
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque
from PIL import ImageDraw, ImageFont, Image
import os
import datetime  # Import for date and time     

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
    st.session_state['wordcloud_history'] = deque(maxlen=10)  # Keep up to 10 wordclouds

# Title
st.title("Ankit's WordCloud App")
st.header("App that takes pdf file as an input, extracts text, preprocess text, removes stop words and build a wordcloud")

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

def get_default_font():
    """Fetch a default truetype font that works across platforms."""
    try:
        return ImageFont.truetype("arial.ttf", 20)
    except IOError:
        return ImageFont.load_default()

def add_watermark(wordcloud_image, text):
    """Add watermark text to a PIL image at the bottom right with a dark grey background."""
    # Get the default font or load a specific one if necessary
    watermark_font = get_default_font()  # Using Default Font
    image = wordcloud_image.convert("RGBA")
    # Create a new RGBA image for the watermark with transparency
    watermark = Image.new("RGBA", image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(watermark)
    # Calculate text size
    text_width, text_height = draw.textbbox((0, 0), text, font=watermark_font)[2:]
    # Position of watermark text in bottom-right corner
    position = (image.width - text_width - 10, image.height - text_height - 10)
    # Draw a dark grey rectangle as the background of the watermark text
    padding = 10  # Padding around the text for the background
    rectangle_position = (position[0] - padding, position[1] - padding, position[0] + text_width + padding, position[1] + text_height + padding)
    draw.rectangle(rectangle_position, fill=(40, 40, 40, 200))  # Dark grey with some transparency
    # Draw the watermark text in bold white color
    draw.text(position, text, font=watermark_font, fill=(255, 255, 255, 255))  # White bold text
    # Combine the watermark with the original image
    combined = Image.alpha_composite(image, watermark)
    return combined.convert("RGB")

# Process uploaded PDF
if uploaded_file is not None:
    processing_message = st.empty()  # Placeholder for processing message
    processing_message.subheader("Processing...")  # Display processing message
    pdf_text = extract_text_from_pdf(uploaded_file)

    if pdf_text:
        processing_message.empty()  # Remove the processing message
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

        # Increment and save global wordcloud count
        global_wordcloud_count += 1
        save_wordcloud_count(global_wordcloud_count)
        st.write(f"WordClouds Generated Globally: {global_wordcloud_count}")

        # Store wordcloud history
        creation_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.wordcloud_history.append((colormap, wordcloud_image, creation_time))

        # Display history pane
        st.subheader("WordCloud History")
        for cmap, wc_img in st.session_state.wordcloud_history:
            st.write(f"Colormap: {cmap}, Created on: {timestamp}")
            st.image(wc_img)
    else:
        processing_message.empty()  # Remove the processing message
        st.error("Unable to extract text from the uploaded PDF.")
else:
    st.info("Please upload a PDF file to proceed.")
