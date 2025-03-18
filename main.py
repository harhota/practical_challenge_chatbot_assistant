import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Load the processed conversations CSV
@st.cache_data
def load_data():
    return pd.read_csv("processed_conversations.csv")

df = load_data()

st.title("Conversation Analysis Dashboard")

# Sidebar filters for conversation type
st.sidebar.header("Filter Conversations")
conversation_type = st.sidebar.selectbox(
    "Select Conversation Type",
    options=["All", "Successful", "Non-Successful", "Outlier (87303 words)"]
)

if conversation_type == "Successful":
    data = df[df["successful"] == True]
elif conversation_type == "Non-Successful":
    data = df[df["successful"] == False]
elif conversation_type == "Outlier (87303 words)":
    data = df[df["dialogue_length"] == 87303]
else:
    data = df

st.write(f"**Number of Conversations:** {len(data)}")

# Display summary metrics if dialogue_length exists
if "dialogue_length" in df.columns:
    avg_length = data["dialogue_length"].mean()
    median_length = data["dialogue_length"].median()
    st.write(f"**Average Dialogue Length (words):** {avg_length:.2f}")
    st.write(f"**Median Dialogue Length (words):** {median_length:.2f}")

# Histogram for dialogue lengths
if "dialogue_length" in df.columns:
    fig = px.histogram(data, x="dialogue_length", nbins=20,
                       title="Distribution of Dialogue Lengths",
                       labels={"dialogue_length": "Dialogue Length (words)"})
    st.plotly_chart(fig)

# Optionally, display details for a specific conversation ID
st.sidebar.header("Conversation Details")
max_id = int(df["conversation_id"].max())
selected_id = st.sidebar.number_input("Enter Conversation ID", min_value=0, max_value=max_id, value=0, step=1)

if st.sidebar.button("Show Conversation Details"):
    conv = df[df["conversation_id"] == selected_id]
    if conv.empty:
        st.write("Conversation not found.")
    else:
        # Show as a dictionary; you could also format this nicely.
        st.write(conv.to_dict(orient="records"))

# Display the system prompt extracted from the dataset
st.header("System Prompt")
try:
    with open("system_prompt.txt", "r", encoding="utf-8") as f:
        system_prompt = f.read()
    st.text_area("System Prompt", value=system_prompt, height=200)
except Exception as e:
    st.write("System prompt file not found.")

# Provide download button for the non-successful conversations DOCX
st.header("Download Documents")
try:
    with open("non_successful_conversations.docx", "rb") as f:
        docx_data = f.read()
    st.download_button(
        label="Download Non-Successful Conversations (DOCX)",
        data=docx_data,
        file_name="non_successful_conversations.docx"
    )
except Exception as e:
    st.write("Non-successful conversations document not available.")

# Display outlier conversation text
st.header("Outlier Conversation Analysis (87303 words)")
try:
    with open("outlier_conversation.txt", "r", encoding="utf-8") as f:
        outlier_text = f.read()
    st.text_area("Outlier Conversation", value=outlier_text, height=300)
except Exception as e:
    st.write("Outlier conversation text file not found.")
