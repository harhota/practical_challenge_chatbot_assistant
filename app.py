import streamlit as st
import pandas as pd
import altair as alt


from funcs import *


# ---------- Data Loading ----------
@st.cache_data(show_spinner=False)
def load_data():
    df = process_conversations("dataset_conversations.txt")
    return df


df = load_data()


# ---------- Section Functions with HTML Anchors ----------

def show_introduction():
    # Add an anchor for scrolling
    st.markdown("<a id='introduction'></a>", unsafe_allow_html=True)
    st.header("1. Introduction")
    st.subheader("Purpose & Context")
    st.markdown("""
    The objective of this analysis is to examine a dataset of conversations from an AI coaching platform.
    We extract actionable insights on user interactions, identify patterns in successful vs. non‑successful conversations,
    and propose improvements to the product.
    """)
    st.subheader("Dataset Overview")
    st.markdown("""
    The dataset is in JSON/JSONL format and contains:
    - Metadata (including error info and success flags),
    - Conversation turns (user, assistant, system),
    - User feedback messages.

    """)


def show_data_processing():
    st.markdown("<a id='data_processing'></a>", unsafe_allow_html=True)
    st.header("2. Data Processing and Methodology")
    st.markdown("""
    **Parsing the Data:**  
    The data is loaded using a function that handles JSONL or JSON arrays. Key fields (metadata, conversation turns,
    final feedback, error information) are extracted.

    **Defining Successful Conversations:**  
    A conversation is flagged as successful if certain heuristics are met (e.g., user sent positive feedback, the 
    conversation was not abrupted). 
    Also there will be some conclusions based on manual human analysis (which sometimes doesn't align with LLM bold evaluation)

    **Data Cleaning & Metrics:**  
    I compute dialogue length (word counts excluding system messages) and turn metrics (user vs. assistant turns,
    average turn length). One extreme outlier (~87,303 words conversation) was identified and removed for realistic statistics - that 
    actually was not a proper dialogue.
    
    There is also separate part in the experiments which leverages LangChain to analyze conversations.
    """)



def show_quant_analysis(df):
    st.markdown("<a id='quant_analysis'></a>", unsafe_allow_html=True)
    st.header("3. Quantitative Analysis")
    st.subheader("Descriptive Statistics at the first glance")

    total_conversations = len(df)
    successful_conversations = df["successful"].sum()
    st.write(f"**Total Conversations:** {total_conversations}")
    st.write(f"**Successful Conversations:** {successful_conversations}")

    metrics_df = pd.DataFrame({
        "Metric": ["Total", "Successful"],
        "Count": [total_conversations, successful_conversations]
    })
    chart = alt.Chart(metrics_df).mark_bar(color="#2ca02c").encode(
        x=alt.X("Metric", sort=None),
        y="Count"
    ).properties()
    st.altair_chart(chart, use_container_width=True)

    st.markdown("### Dialogue Length Metrics (Without Outlier)")
    if "dialogue_length" in df.columns:
        # Remove the outlier (assumes the max value is the outlier)
        max_length = df["dialogue_length"].max()
        df_no_outlier = df[df["dialogue_length"] != max_length]
        avg_length = df_no_outlier["dialogue_length"].mean()
        median_length = df_no_outlier["dialogue_length"].median()

        st.metric("Average Dialogue Length", f"{int(avg_length)} words")
        st.metric("Median Dialogue Length", f"{int(median_length)} words")

        lengths_df = pd.DataFrame({
            "Metric": ["Average", "Median"],
            "Words": [int(avg_length), int(median_length)]
        })
        chart2 = alt.Chart(lengths_df).mark_bar(color="#ff7f0e").encode(
            x=alt.X("Metric", sort=None),
            y="Words"
        ).properties(title="Dialogue Length Metrics")
        st.altair_chart(chart2, use_container_width=True)
    else:
        st.error("The 'dialogue_length' column is missing from the data.")

    if "turn_metrics" in df.columns:
        # Compute median of word counts per turn for each conversation
        df["median_turn_length"] = df["turn_metrics"].apply(
            lambda metrics: int(np.median(metrics.get("words_per_turn", []))) if metrics.get("words_per_turn",
                                                                                             []) else 0
        )
    else:
        # Fallback: if "turn_metrics" doesn't exist, use "dialogue_length"
        # (This is less granular though.)
        df["median_turn_length"] = df["dialogue_length"]

    st.subheader("Median Turn Length per Conversation")
    # Build a DataFrame with conversation id and median turn length, sorted by conversation id
    median_df = compute_median_dialogue_lengths(df, outlier_conversation_id=0)
    st.bar_chart(median_df.set_index("conversation_id"))

    st.markdown("""
    **Turn Balance:**
Most conversations show a balanced turn distribution between the user and the assistant. For instance, in almost every dialogue, the number of user turns and assistant turns is almost equal, which may indicate that both parties are engaged in a back-and-forth exchange. This balance could be a positive indicator of a collaborative conversation.

**Conversation Length Variation:**
The total turn counts vary widely—from as few as 13 turns (Conversation 12) to as many as 43 turns (Conversation 14). Longer conversations (in terms of turns or total words) might suggest more in‑depth or complex discussions, while shorter ones could be more straightforward or focused.

**Average Turn Length Differences:**
Average turn lengths range from about 51 words (Conversation 17) to over 110 words (Conversation 3). Longer turns may *indicate more detailed or explanatory responses*, whereas shorter turns might be more to the point.

For example, **Conversation 3**’s higher average (110.80 words per turn) could reflect a more elaborative or detailed exchange, perhaps tackling a more nuanced issue.
In contrast, **Conversation 17**’s lower average (51.48 words per turn) might suggest a brisk, succinct conversation style.

**Implications for Dialogue Quality:**
A balanced turn-taking structure generally indicates that both sides are contributing. However, whether longer or shorter turns lead to higher user satisfaction **might depend on the context**:

In some situations, concise answers (shorter turns) could be preferred for efficiency.
In other contexts, more detailed answers (longer turns) might be necessary to cover complex topics.

**Potential for Further Analysis:**
It could be useful to correlate these turn metrics with other factors (e.g., user feedback or outcome measures) to see if, for example, conversations with a particular range of average turn lengths tend to be rated more highly. Also, analyzing whether longer conversations tend to be more engaging or if they sometimes indicate over-elaboration could provide deeper insights.
    """)
    # (Optional) You can also display the DataFrame as a table:
    # st.write("Detailed median turn lengths:", median_df)

# def show_qual_analysis():
#     st.markdown("<a id='qual_analysis'></a>", unsafe_allow_html=True)
#     st.header("4. Qualitative Analysis")
#     st.markdown("""
#     **Thematic Insights:**
#     Common themes include challenges with providing constructive feedback, negativity in user interactions,
#     and extended dialogues in non‑successful conversations.
#
#
#     **LangChain Reflections:**
#     Initial attempts using LangChain to categorize conversations (e.g., 'Conflict Resolution' or 'Feedback Handling')
#     produced verbose or ambiguous outputs. Refining prompts or exploring alternative NLP methods may yield clearer insights.
#
#     **Human notes:**
#     Actually the most of dialogues make a lot of sense!
#
#     Personally I managed to look through 7-8 dialogues and read them carefully - they are pretty long and assistant
#     response in a conscious way, understanding the context well and
#     providing absolutely reasonable ideas. Also most of the time assistant is really leading to solving the problem,
#     not just chatting for fun.
#     """)

def show_qual_analysis():
    st.header("Qualitative Analysis")

    st.markdown("### Conversation-Specific Qualitative Insights")

    # Conversation #3: Highest average words per turn
    conv3 = df[df["conversation_id"] == 3]
    if not conv3.empty:
        with st.expander("Conversation #3 Analysis (Highest Average Words per Turn)"):
            st.markdown("""
            #### **Analyzing Conversation #3**
            **Summary:**  
            In this conversation, the user prepares for a difficult discussion with a client. The context is that the user's team is undergoing a restructuring—because the business is moving to a more cost‑effective country, the team needs to meet a specific Leader-to-team member ratio, which now renders one Leader surplus. The user explains that one Leader, who has demonstrated strong technical skills and contributed significantly to the team, might be re‑assigned to a new role. However, the client is known to be extremely cost‑focused and numbers‑oriented and is new to this line of business, meaning that trust hasn’t been fully established yet.

            Throughout the dialogue, the assistant guides the user in clarifying not only the factual background (such as the current ratio 
            and the observed technical proficiency) but also the **deeper concerns and unexpressed feelings**. The user reveals internal conflicts: on one hand, 
            a sense of responsibility to advocate for the team and explore all options for retaining valuable talent; on the other, a worry about being perceived as 
            misaligned with the client’s cost‑cutting strategy or as incompetent.

            **Conclusions and Relevance:**  
            This **conversation is highly relevant**  because it captures both the logistical challenges and the emotional complexities involved in a strategic business decision. The dialogue reflects a balanced approach to addressing cost priorities while safeguarding valuable human capital, highlighting the importance of empathy, active listening, and establishing trust with a cost‑focused client.
            """)
    else:
        st.write("Conversation #3 not found in the dataset.")

    # Conversation #17: Lowest average words per turn
    conv17 = df[df["conversation_id"] == 17]
    if not conv17.empty:
        with st.expander("Conversation #17 Analysis (Lowest Average Words per Turn)"):
            st.markdown("""
            #### **Analyzing Conversation #17**
            **Summary:**  
            In this conversation, the dialogue is **noticeably concise**, with a much lower average word count per turn. This brevity 
            may indicate a more focused or perhaps rushed exchange. While the exact details of the discussion can vary, such succinct 
            interactions might suggest that either the user or the assistant (or both) are providing very direct responses without 
            delving deeply into the underlying issues. This could be beneficial when time is short but might also risk missing nuanced aspects of the conversation.

            **Conclusions and Relevance:**  
            The short average turn length suggests a **different communication style** compared to longer, more elaborative dialogues. 
            In this context, the challenge may lie in ensuring that essential details and emotional nuances are still effectively 
            communicated despite the brevity. This observation could prompt further investigation into whether the concise format 
            serves the conversation’s purpose or if it leads to potential misunderstandings.
            """)
    else:
        st.write("Conversation #17 not found in the dataset.")

    st.markdown("### Overall Qualitative Reflections")
    st.markdown("""
    **Thematic Insights:**  
    - Many conversations reveal challenges in handling feedback and addressing underlying emotions, with successful dialogues generally showing clear, concise user feedback.  
    - In non‑successful interactions, the lack of brief, actionable feedback is often noticeable, and some dialogues exhibit excessive verbosity (as seen in the outlier).

    **LangChain Reflections:**  
    - Initial attempts to use LangChain for qualitative categorization and response evaluation yielded overly verbose results, indicating that a more refined prompt or alternative techniques (such as sentiment analysis or topic modeling) might be needed to extract more actionable insights.

    **Implications for Product Improvement:**  
    - Deeper analysis of suddenly abrupted conversations (what are the reasons)
    - Refine feedback detection heuristics, possibly integrating more advanced NLP techniques.  
    

    **Recommendations:**  
    - Continue using user feedback loops to continuously improve the assistant’s responses.
    - Implement additional qualitative filters to identify key recurring topics and potential communication breakdowns.  
    """)




def show_conclusions():
    st.markdown("<a id='conclusions'></a>", unsafe_allow_html=True)
    st.header("5. Conclusions and Recommendations")
    st.markdown("""
    **Key Findings:**  
    - Most conversations have good quality and users are totally fine (I would say 4/5) with the agent reponses
    - Clear and actionable user feedback correlates with successful conversations (I would rate those 5/5).

    **Recommendations:**  
    1. **Data Handling:** Flag and further investigate overly long dialogues.
    2. **Heuristic Refinement:** Incorporate advanced NLP and metrics (there is a need to dive into some papers)
     to better determine conversation success. Also RLHF can be used for implementing some reward function.
    3. **System Enhancements:** Continue using user feedback loops during the conversation.
    4. **Future Analysis:** Consider alternative AI analysis frameworks or refine LangChain prompts.
    """)




# ---------- Main Function with Two Navigation Methods ----------
def main():
    # Set the page configuration with a dark theme (you can tweak this in the config file as well)
    st.set_page_config(page_title="Conversation Analysis Report", layout="wide", initial_sidebar_state="expanded")

    # Sidebar Navigation using clickable links (for auto-scroll) and a radio button
    st.sidebar.title("Navigation")

    #st.sidebar.markdown("### Jump to Section")
    # Clickable links (auto-scroll)
    st.sidebar.markdown("[Introduction](#introduction)", unsafe_allow_html=True)
    st.sidebar.markdown("[Data Processing](#data_processing)", unsafe_allow_html=True)
    st.sidebar.markdown("[Quantitative Analysis](#quant_analysis)", unsafe_allow_html=True)
    st.sidebar.markdown("[Qualitative Analysis](#qual_analysis)", unsafe_allow_html=True)
    st.sidebar.markdown("[Conclusions & Recommendations](#conclusions)", unsafe_allow_html=True)

    # st.sidebar.markdown("### Or Select Section")
    # nav_option = st.sidebar.radio("Section:", [
    #     "Introduction",
    #     "Data Processing",
    #     "Quantitative Analysis",
    #     "Qualitative Analysis",
    #     "Discussion",
    #     "Conclusions & Recommendations",
    #     "Appendices"
    # ])

    # Also show the system prompt
    st.sidebar.header("Assistant System Prompt")
    try:
        with open("system_prompt.txt", "r", encoding="utf-8") as f:
            system_prompt = f.read()
        st.sidebar.text_area("always the same", value=system_prompt, height=200)
    except Exception as e:
        st.sidebar.error(f"Error reading system_prompt.txt: {e}")

    show_introduction()
    show_data_processing()
    show_quant_analysis(df)
    show_qual_analysis()
    show_conclusions()




if __name__ == "__main__":
    main()
