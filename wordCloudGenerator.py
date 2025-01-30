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
import firebase_admin
from firebase_admin import credentials, firestore
from google.oauth2 import id_token
from google_auth_oauthlib.flow import Flow
from streamlit.components.v1 import html

# Configuration
COUNT_FILE = "wordcloud_count.txt"
LOCK_FILE = f"{COUNT_FILE}.lock"
CACHE_TIMEOUT = 3600  # 1 hour
HISTORY_LENGTH = 10

# Initialize Firebase
# Modified Firebase initialization section
try:
    # Initialize Firebase only once
    if not firebase_admin._apps:
        firebase_cred = credentials.Certificate(st.secrets["firebase"]["credential"])
        firebase_app = firebase_admin.initialize_app(firebase_cred)
        db = firestore.client()
        st.success("Firebase initialized successfully!")
    else:
        db = firestore.client()
except ValueError as ve:
    st.error(f"Firebase initialization error: {str(ve)}")
    db = None
except KeyError as ke:
    st.error(f"Missing Firebase credentials in secrets: {str(ke)}")
    db = None
except Exception as e:
    st.error(f"Firebase connection failed: {str(e)}")
    db = None

# Read the secret key named "ANKIT_SECRET" from Streamlit secrets
if 'SECRETWORD' in st.secrets:
    ANKIT_SECRET = st.secrets['SECRETWORD']
    SECRET_HASH = hashlib.sha256(ANKIT_SECRET.encode()).hexdigest()
else:
    st.error("Secret Key not found!")

# Authentication functions
def get_google_flow():
    return Flow.from_client_config(
        client_config={
            "web": {
                "client_id": st.secrets["google_oauth"]["client_id"],
                "client_secret": st.secrets["google_oauth"]["client_secret"],
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://accounts.google.com/o/oauth2/token",
                "redirect_uris": [st.secrets["google_oauth"]["redirect_uri"]]
            }
        },
        scopes=[
            "https://www.googleapis.com/auth/userinfo.profile",
            "https://www.googleapis.com/auth/userinfo.email",
            "openid"
        ]
    )

def verify_token(token):
    try:
        return id_token.verify_oauth2_token(token, requests.Request(), st.secrets["google_oauth"]["client_id"])
    except ValueError:
        return None

def get_current_user():
    return st.session_state.get("user")

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
    except Exception as e:
        st.error(f"Error saving count: {str(e)}")

@st.cache_data(ttl=CACHE_TIMEOUT)
def get_default_font(size=20):
    """Load font with fallback strategy"""
    try:
        return ImageFont.truetype("arial.ttf", size)
    except IOError:
        try:
            return ImageFont.truetype("LiberationSans-Regular.ttf", size)
        except IOError:
            try:
                return ImageFont.truetype("DejaVuSans.ttf", size)
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
st.title("📚 Ankit's WordCloud: Advanced Text Visualization")
st.subheader("App that takes PDF file as input, extracts text, preprocesses it, removes stop words, and builds a word cloud and now can generate an animated version of it too")
st.markdown("---")

# ======================
# Sidebar Configuration
# ======================
with st.sidebar:
    st.header("Settings")
    
    # Authentication Section
    with st.expander("Account"):
        if 'user' not in st.session_state:
            flow = get_google_flow()
            auth_url, _ = flow.authorization_url(prompt='consent')
            
            html(f"""
                <a href="{auth_url}" target="_self" style="
                    background: #4285f4;
                    color: white;
                    padding: 10px 20px;
                    border-radius: 5px;
                    text-decoration: none;
                    display: inline-block;
                ">
                    Sign in with Google
                </a>
            """, height=50)
            
            if st.query_params.get("code"):
                try:
                    flow.fetch_token(code=st.query_params["code"])
                    credentials = flow.credentials
                    user_info = verify_token(credentials.id_token)
                    
                    if user_info:
                        st.session_state.user = {
                            "uid": user_info["sub"],
                            "name": user_info.get("name", ""),
                            "email": user_info.get("email", ""),
                            "photo": user_info.get("picture", ""),
                            "gender": ""
                        }
                        
                        user_ref = db.collection("users").document(user_info["sub"])
                        user_ref.set({
                            "name": user_info.get("name", ""),
                            "email": user_info.get("email", ""),
                            "photo_url": user_info.get("picture", ""),
                            "created_at": firestore.SERVER_TIMESTAMP,
                            "last_login": firestore.SERVER_TIMESTAMP,
                            "gender": "",
                            "history": []
                        }, merge=True)
                        
                        st.experimental_set_query_params()
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"Authentication failed: {str(e)}")
        
        else:
            user = st.session_state.user
            st.image(user['photo'], width=100)
            st.write(f"**{user['name']}**")
            st.write(user['email'])
            
            new_gender = st.selectbox(
                "Your Gender",
                ["", "Male", "Female", "Other"],
                index=["", "Male", "Female", "Other"].index(user['gender'])
            )
            if new_gender != user['gender']:
                db.collection("users").document(user['uid']).update({"gender": new_gender})
                st.session_state.user['gender'] = new_gender
                st.success("Gender updated!")
            
            if st.button("Sign Out"):
                del st.session_state.user
                st.rerun()
    
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
    text = re.sub(r'\d+', '', text) if remove_numbers else text
    text = re.sub(r'[^\w\s]', '', text) if remove_punctuation else text
    text = re.sub(r'[^\w\s-]|(?<!\w)-(?!\w)', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    if user_stopwords:
        custom_stops = {word.strip().lower() for word in user_stopwords.split(',')}
        st.session_state.custom_stopwords.update(custom_stops)
    
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
    wc = None
    gif_buffer = None

    try:
        if not isinstance(text, str):
            st.error(f"Invalid text type received: {type(text)}")
            return None, None

        with st.spinner("Cleaning text..."):
            cleaned_text = preprocess_text(text)
            progress_bar.progress(20)

            if not cleaned_text.strip():
                st.error("No valid text remaining after preprocessing.")
                return None, None

        mask = generate_mask(mask_image) if mask_image else None
        progress_bar.progress(40)

        font_path = None
        if custom_font:
            try:
                with open("temp_font.ttf", "wb") as f:
                    f.write(custom_font.getbuffer())
                font_path = "temp_font.ttf"
            except Exception as e:
                st.error(f"Failed to load custom font: {str(e)}")
                font_path = None

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
            font_path=font_path
        )

        with st.spinner("Generating visualization..."):
            validation_text = cleaned_text.strip()
            if len(validation_text) < 10:
                st.error(f"Insufficient text ({len(validation_text)} characters)")
                return None, None

            wc.generate(validation_text)
            progress_bar.progress(80)

        if animate_wc:
            with st.spinner("Rendering animation..."):
                frames = []
                for _ in range(15):
                    temp_img = Image.new("RGB", wc.to_image().size, (255, 255, 255))                    
                    draw = ImageDraw.Draw(temp_img)
                    for (word, size, pos, _, color) in wc.layout_:
                        x, y = pos[0] + random.randint(-5,5), pos[1] + random.randint(-5,5)
                        font = get_default_font(size)
                        draw.text((x, y), word[0], font=font, fill=color)
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

    except ValueError as ve:
        st.error(f"Validation Error: {str(ve)}")
        return None, None
    except Exception as e:
        st.error(f"Unexpected Error: {str(e)}")
        return None, None

# ======================
# Main Application Flow
# ======================
uploaded_file = st.file_uploader("Upload PDF Document", type="pdf")

if uploaded_file:
    wordcloud = None
    animation = None
    
    with st.spinner("Extracting text..."):
        pdf_reader = PdfReader(uploaded_file)
        text = "\n".join([page.extract_text() or "" for page in pdf_reader.pages])
    
    if text:
        result = generate_wordcloud(text)
        
        if result is not None:
            wordcloud, animation = result
        else:
            st.error("Failed to generate word cloud.")
        
        if wordcloud:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Word Cloud")
                wc_image = wordcloud.to_image()
                if not hashlib.sha256(secret_access.encode()).hexdigest() == SECRET_HASH:
                    wc_image = add_watermark(wc_image, "Generated @ Ankit's WordCloud")
                st.image(wc_image, use_container_width=True)
                
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
                    st.image(animation.getvalue(), use_container_width=True)
                    st.download_button(
                        label="Download Animation",
                        data=animation.getvalue(),
                        file_name="wordcloud.gif",
                        mime="image/gif"
                    )

            try:
                global_count = load_wordcloud_count() + 1
                save_wordcloud_count(global_count)
            except Exception as e:
                st.error(f"Failed to update global count: {str(e)}")
        
            if get_current_user():
                user_ref = db.collection("users").document(st.session_state.user['uid'])
                user_ref.update({
                    "history": firestore.ArrayUnion([{
                        "timestamp": datetime.datetime.now().isoformat(),
                        "config": {
                            "colormap": colormap,
                            "max_words": max_words,
                            "mask_used": bool(mask_image)
                        }
                    }])
                })

            st.session_state.wordcloud_history.append({
                "timestamp": datetime.datetime.now().isoformat(),
                "image": wc_image,
                "config": {
                    "colormap": colormap,
                    "max_words": max_words,
                    "mask_used": bool(mask_image)
                }
            })

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
