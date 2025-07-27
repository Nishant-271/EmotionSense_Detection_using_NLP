import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib

# Load the trained model
pipe_lr = joblib.load(open("model/text_emotion.pkl", "rb"))

# Emoji dictionary
emotions_emoji_dict = {
    "anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗", "joy": "😂",
    "neutral": "😐", "sad": "😔", "sadness": "😔", "shame": "😳", "surprise": "😮"
}

# Prediction functions
def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]

def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results

# Streamlit app
def main():
    # Custom CSS Styling
    st.markdown("""
        <style>
        html, body, [class*="css"]  {
            font-family: 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #fefcea, #f1daff);
            color: #1a1a40;
        }

        .main-title {
            font-size: 2.7rem;
            font-weight: bold;
            text-align: center;
            padding: 1rem;
            margin-bottom: 1rem;
            border-radius: 1rem;
            background: linear-gradient(to right, #fefcea, #f1daff);
            color: #1a1a40;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }

        .subheader {
            font-size: 1.5rem;
            font-weight: 600;
            text-align: center;
            color: #333;
            margin-bottom: 1rem;
        }

        .stTextArea textarea {
            font-size: 1.2rem;
            background-color: #f8f8fa;
            border-radius: 0.5rem;
        }

        .stButton>button {
            background: linear-gradient(to right, #a1c4fd, #c2e9fb);
            color: black;
            font-weight: bold;
            font-size: 1.1rem;
            padding: 0.5rem 1.2rem;
            border: none;
            border-radius: 10px;
        }

        .stButton>button:hover {
            background: linear-gradient(to right, #89f7fe, #66a6ff);
            color: white;
            transition: 0.3s ease-in-out;
        }

        .emoji-result {
            font-size: 1.7rem;
            font-weight: bold;
        }
        </style>
    """, unsafe_allow_html=True)

    # Title and Subheader
    st.markdown('<div class="main-title">💬 Text Emotion Detection App</div>', unsafe_allow_html=True)
    st.markdown('<div class="subheader">Detect the Emotion Behind Any Text Instantly 🔍</div>', unsafe_allow_html=True)

    with st.form(key='my_form'):
        raw_text = st.text_area("📝 Type your text here:")
        submit_text = st.form_submit_button(label='🔍 Submit')

    if submit_text:
        col1, col2 = st.columns(2)

        prediction = predict_emotions(raw_text)
        probability = get_prediction_proba(raw_text)

        with col1:
            st.success("✅ Original Text")
            st.write(raw_text)

            st.success("🎯 Prediction")
            emoji_icon = emotions_emoji_dict[prediction]
            st.markdown(f'<div class="emoji-result">{prediction} {emoji_icon}</div>', unsafe_allow_html=True)

            st.markdown(f"**Confidence Score:** `{np.max(probability):.2f}`")

        with col2:
            st.success("📊 Prediction Probability")
            proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
            proba_df_clean = proba_df.T.reset_index()
            proba_df_clean.columns = ["emotions", "probability"]

            fig = alt.Chart(proba_df_clean).mark_bar().encode(
                x=alt.X('emotions', sort='-y'),
                y='probability',
                color='emotions'
            ).properties(
                width=350,
                height=300
            )
            st.altair_chart(fig, use_container_width=True)

if __name__ == '__main__':
    main()
