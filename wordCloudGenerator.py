import streamlit as st
from wordcloud import WordCloud, STOPWORDS
from PyPDF2 import PdfReader
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError
import numpy as np
import requests
from io import BytesIO
import os
import datetime
import streamlit as st

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

def get_default_font():
    """Fetch a default truetype font that works across platforms."""
    try:
        return ImageFont.truetype("arial.ttf", 20)
    except IOError:
        return ImageFont.load_default()

def add_watermark(wordcloud_image, text):
    """Add watermark text to a PIL image at the bottom right with a dark grey background."""
    watermark_font = get_default_font()
    image = wordcloud_image.convert("RGBA")
    watermark = Image.new("RGBA", image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(watermark)
    text_width, text_height = draw.textbbox((0, 0), text, font=watermark_font)[2:]
    position = (image.width - text_width - 10, image.height - text_height - 10)
    padding = 10
    rectangle_position = (position[0] - padding, position[1] - padding, position[0] + text_width + padding, position[1] + text_height + padding)
    draw.rectangle(rectangle_position, fill=(40, 40, 40, 200))
    draw.text(position, text, font=watermark_font, fill=(255, 255, 255, 255))
    combined = Image.alpha_composite(image, watermark)
    return combined.convert("RGB")

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
# Updated company logos with PNG URLs
company_logos = {
    "Default": None,
    "Google": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4a/Logo_2013_Google.png/512px-Logo_2013_Google.png",
    "Microsoft": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/44/Microsoft_logo.svg/512px-Microsoft_logo.svg.png",
    "Apple": "https://upload.wikimedia.org/wikipedia/commons/thumb/f/fa/Apple_logo_black.svg/512px-Apple_logo_black.svg.png",
    "Amazon": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a9/Amazon_logo_black.svg/512px-Amazon_logo_black.svg.png"
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
    headers = {
        'User-Agent': 'YourAppName/1.0 (your-email@example.com)'
    }
    try:
        if logo_url:
            response = requests.get(logo_url, headers=headers, stream=True, timeout=10)
            response.raise_for_status()  # Raise an error for failed requests
            logo_image = Image.open(BytesIO(response.content)).convert("1")
            mask_array = np.array(convert_white_to_transparent(logo_image))
            # Check the shape of the mask
            if len(mask_array.shape) != 2:
                raise ValueError("Mask image must be a 2D array")
            # Create a binary mask where black pixels are 1 and white pixels are 0
            binary_mask = np.where(mask_array == 0, 1, 0)
        return binary_mask
    except requests.RequestException as e:
        st.warning(f"Failed to load the logo image due to a network error: {e}. Defaulting to no mask.")
    except UnidentifiedImageError as e:
        st.warning(f"Failed to process the logo image: {e}. Defaulting to no mask.")
    return None
    
def convert_white_to_transparent(image):
    # Get the data of the image
    data = image.getdata()
    
    # Create a new data list
    new_data = []
    
    for item in data:
        # Change all white (also shades of whites)
        # pixels to transparent
        if item[:3] == (255, 255, 255):
            new_data.append((255, 255, 255, 0))
        else:
            new_data.append(item)
    
    # Update image data
    image.putdata(new_data)
    
    # return the image
    return image

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
