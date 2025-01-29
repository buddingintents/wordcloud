import streamlit as st
from wordcloud import WordCloud, STOPWORDS
from PyPDF2 import PdfReader
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError
from io import BytesIO
import requests
import re
import hashlib
import os
import datetime
import random
from filelock import FileLock, Timeout
from collections import deque

# Configuration
COUNT_FILE = "wordcloud_count.txt"
LOCK_FILE = f"{COUNT_FILE}.lock"
CACHE_TIMEOUT = 3600  # 1 hour
HISTORY_LENGTH = 10

# Security Configuration
SECRET_HASH = hashlib.sha256(b'default_secret').hexdigest()  # Set via secrets in production

@st.cache_data(ttl=CACHE_TIMEOUT)
def load_wordcloud_count():
    """Safely load global word cloud count with file locking"""
    try:
        with FileLock(LOCK_FILE, timeout=3):
            if os.path.exists(COUNT_FILE):
                with open(COUNT_FILE, "r") as f:
                    return int(f.read().strip())
            return 0
    except Timeout:
        st.error("Another instance is updating the count. Please try again.")
        return 0

def save_wordcloud_count(count):
    """Safely update global word cloud count with file locking"""
    try:
        with FileLock(LOCK_FILE, timeout=3):
            with open(COUNT_FILE, "w") as f:
                f.write(str(count))
    except Timeout:
        st.error("Couldn't update count due to concurrent access.")

@st.cache_data(ttl=CACHE_TIMEOUT)
def get_default_font(size=20):
    """Load font with fallback strategy"""
    try:
        return ImageFont.truetype("arial.ttf", size)
    except IOError:
        try:
            return ImageFont.truetype("LiberationSans-Regular.ttf", size)
        except IOError:
            return ImageFont.load_default(size)

def add_watermark(image, text):
    """Add dynamic watermark with responsive positioning"""
    draw = ImageDraw.Draw(image.convert("RGBA"))
    try:
        font = get_default_font(16)
    except IOError:
        font = ImageFont.load_default(16)
    
    text_width, text_height = draw.textbbox((0, 0), text, font=font)[2:]
    padding = 10
    position = (
        image.width - text_width - padding,
        image.height - text_height - padding
    )
    
    # Create semi-transparent background
    overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle(
        [position[0]-padding, position[1]-padding,
         position[0]+text_width+padding, position[1]+text_height+padding],
        fill=(40, 40, 40, 180)
    )
    overlay_draw.text(position, text, font=font, fill=(255, 255, 255, 240))
    
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")

# Initialize session state
if 'wordcloud_history' not in st.session_state:
    st.session_state.wordcloud_history = deque(maxlen=HISTORY_LENGTH)
if 'custom_stopwords' not in st.session_state:
    st.session_state.custom_stopwords = set()

# UI Configuration
st.set_page_config(page_title="WordCloud Pro", layout="wide")
st.title("📚 WordCloud Pro: Advanced Text Visualization")
st.markdown("---")

# ======================
# Sidebar Configuration
# ======================
with st.sidebar:
    st.header("Settings")
    
    with st.expander("Visual Style"):
        colormap = st.selectbox("Color Map", sorted(plt.colormaps()), index=112)
        background_color = st.color_picker("Background Color", "#FFFFFF")
        max_words = st.slider("Max Words", 100, 1000, 500, 50)
        animate_wc = st.checkbox("Generate Animation", True)
    
    with st.expander("Text Processing"):
        user_stopwords = st.text_area("Add Custom Stopwords (comma-separated)", 
                                    help="Example: company, product, trademark")
        remove_numbers = st.checkbox("Remove Numbers", True)
        remove_punctuation = st.checkbox("Remove Punctuation", True)
    
    with st.expander("Advanced"):
        secret_access = st.text_input("Secret Access Key", type="password")
        custom_font = st.file_uploader("Custom Font (TTF)", type=["ttf", "otf"])
        mask_image = st.file_uploader("Custom Mask Image", type=["png", "jpg", "jpeg"])

# ======================
# Main Processing Logic
# ======================
def preprocess_text(text):
    """Enhanced text cleaning with multiple options"""
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    
    # Process custom stopwords
    """if user_stopwords:
        custom_stops = {word.strip().lower() for word in user_stopwords.split(',')}
        st.session_state.custom_stopwords.update(custom_stops)
    """
    return text.lower()

def generate_mask(uploaded_image):
    """Process user-uploaded mask image"""
    try:
        img = Image.open(uploaded_image).convert("RGBA")
        grayscale = np.array(img.convert("L"))
        threshold = 128
        mask = 255 - (grayscale > threshold) * 255
        return mask.astype(np.int_)
    except Exception as e:
        st.error(f"Mask processing error: {str(e)}")
        return None

def generate_wordcloud(text):
    """Main wordcloud generation with progress tracking"""
    progress_bar = st.progress(0)
    wc = None  # Initialize wc variable
    
    try:
        # Validate input before processing
        if not isinstance(text, str):
            st.error(f"Invalid text type received: {type(text)}. Expected string.")
            return None, None

        # Preprocess text with enhanced validation
        with st.spinner("Cleaning text..."):
            cleaned_text = preprocess_text(text)
            progress_bar.progress(20)

            if not cleaned_text.strip():
                st.error("No valid text remaining after preprocessing. Check your filters and stopwords.")
                return None, None

        # Generate mask
        mask = None
        if mask_image:
            with st.spinner("Processing mask..."):
                mask = generate_mask(mask_image) if mask_image else None
                progress_bar.progress(40)

        # Create wordcloud instance BEFORE generation
        wc = WordCloud(
            width=1200,
            height=600,
            max_words=max_words,
            colormap=colormap,
            background_color=background_color,
            stopwords=STOPWORDS.union(st.session_state.custom_stopwords),
            mask=mask,
            contour_width=2,
            contour_color=background_color,
            font_path=custom_font.name if custom_font else None
        )

        with st.spinner("Generating visualization..."):
            # Final validation before generation
            validation_text = cleaned_text.strip()
            if len(validation_text) < 10:
                st.error(f"Insufficient text for generation ({len(validation_text)} characters)")
                return None, None

            wc.generate(validation_text)  # Now wc is defined
            progress_bar.progress(80)

    except ValueError as ve:
        st.error(f"Validation Error: {str(ve)}")
        return None, None
    except TypeError as te:
        st.error(f"Type Error: {str(te)}")
        st.error(f"Problematic text type: {type(cleaned_text)}")
        return None, None
    except Exception as e:
        st.error(f"Unexpected Error: {str(e)}")
        st.error("Please check terminal for full error details")
        raise e  # Re-raise to see full stack trace
        
        # Generate animation if enabled
        if animate_wc:
            with st.spinner("Rendering animation..."):
                frames = []
                for _ in range(10):  # Reduced frame count for performance
                    temp_img = wc.to_image()
                    draw = ImageDraw.Draw(temp_img)
                    for (word, size, pos, _, color) in wc.layout_:
                        x, y = pos[0] + random.randint(-5,5), pos[1] + random.randint(-5,5)
                        font = get_default_font(size)
                        draw.text((x, y), str(word), font=font, fill=color)
                    frames.append(temp_img)
                
                gif_buffer = BytesIO()
                frames[0].save(
                    gif_buffer,
                    format="GIF",
                    save_all=True,
                    append_images=frames[1:],
                    duration=300,
                    loop=0
                )
                progress_bar.progress(100)
        
        return wc, gif_buffer if animate_wc else None
    except Exception as e:
        st.error(f"Generation failed: {str(e)}")
        return None, None  # Always return a tuple

# ======================
# Main Application Flow
# ======================
uploaded_file = st.file_uploader("Upload PDF Document", type="pdf")

if uploaded_file:
    # Initialize variables with default values
    wordcloud = None
    animation = None
    # Extract text from PDF
    with st.spinner("Extracting text..."):
        pdf_reader = PdfReader(uploaded_file)
        text = "\n".join([page.extract_text() or "" for page in pdf_reader.pages])
    
    if text:
         # Generate visualization with null check
        result = generate_wordcloud(text)
        
        if result is not None:
            wordcloud, animation = result
            # Rest of display code...
        else:
            st.error(text)
            st.error("Failed to generate word cloud. Please check input and settings.")
        
        if wordcloud:
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Word Cloud")
                wc_image = wordcloud.to_image()
                if not hashlib.sha256(secret_access.encode()).hexdigest() == SECRET_HASH:
                    wc_image = add_watermark(wc_image, "Generated with WordCloud Pro")
                st.image(wc_image, use_column_width=True)
                
                # Download buttons
                img_buffer = BytesIO()
                wc_image.save(img_buffer, format="PNG")
                st.download_button(
                    label="Download Image",
                    data=img_buffer.getvalue(),
                    file_name="wordcloud.png",
                    mime="image/png"
                )
            
            with col2:
                if animation:
                    st.subheader("Animation Preview")
                    st.image(animation.getvalue(), use_column_width=True)
                    st.download_button(
                        label="Download Animation",
                        data=animation.getvalue(),
                        file_name="wordcloud.gif",
                        mime="image/gif"
                    )
            
            # Update global count and history
            global_count = load_wordcloud_count() + 1
            save_wordcloud_count(global_count)
            
            st.session_state.wordcloud_history.append({
                "timestamp": datetime.datetime.now().isoformat(),
                "image": wc_image,
                "config": {
                    "colormap": colormap,
                    "max_words": max_words,
                    "mask_used": bool(mask_image)
                }
            })
            
            # Display history
            with st.expander("Generation History"):
                for idx, entry in enumerate(st.session_state.wordcloud_history):
                    cols = st.columns([1, 4])
                    cols[0].image(entry["image"], width=150)
                    cols[1].write(f"**Generation #{idx+1}**")
                    cols[1].write(f"Date: {entry['timestamp']}")
                    cols[1].write(f"Color Map: {entry['config']['colormap']}")
            
            st.success(f"Successfully generated! Total visualizations created: {global_count}")
    else:
        st.error("No extractable text found in the document.")
