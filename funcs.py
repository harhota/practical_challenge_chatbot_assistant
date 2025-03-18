# funcs.py

import json
import os
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize


nltk.download('punkt')
nltk.download('punkt_tab')


def is_successful(conversation, last_n=5, feedback_length_threshold=50):
    """
    Determine if a conversation is successful by checking if:
      - One of the last `last_n` messages contains the word "feedback"
      - At least one user message among those last messages is short enough (assumed to be feedback)
    Returns a tuple (success_flag, feedback_message).
    """
    messages = conversation.get("inputs", {}).get("messages", [])
    if len(messages) < 3:
        return False, None  # Too short to be a proper conversation

    last_messages = messages[-last_n:]
    if not any("feedback" in msg.get("content", "").lower() for msg in last_messages):
        return False, None

    user_messages = [msg.get("content", "").strip() for msg in last_messages if msg.get("role", "").lower() == "user"]
    feedback_candidates = [msg for msg in user_messages if len(msg) < feedback_length_threshold]

    if feedback_candidates:
        return True, feedback_candidates[-1]

    return False, None

def compute_dialogue_length(messages):
    """
    Compute the total number of words in a conversation's messages,
    excluding messages from the 'system' role.
    """
    total_words = 0
    for msg in messages:
        if msg.get("role", "").lower() == "system":
            continue
        content = msg.get("content", "")
        total_words += len(word_tokenize(content))
    return total_words

def process_conversations(file_path):
    """
    Reads and processes the dataset file.
    Returns a DataFrame with one row per conversation including:
      - metadata
      - messages
      - final feedback
      - success flag
      - error_info
      - dialogue_length (computed from the messages)
    """
    conversations = []

    # Determine file type: JSON array vs. JSONL
    with open(file_path, "r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == '[':
            data = json.load(f)
        else:
            data = [json.loads(line) for line in f if line.strip()]

    print(f"Total conversations found: {len(data)}")

    for idx, conv in enumerate(data):
        metadata = conv.get("metadata", {})
        inputs = conv.get("inputs", {})
        messages = inputs.get("messages", [])
        success, feedback = is_successful(conv, last_n=5, feedback_length_threshold=50)
        error_info = metadata.get("error", None)

        conversation_entry = {
            "conversation_id": idx,
            "metadata": metadata,
            "messages": messages,
            "final_feedback": feedback,
            "successful": success,
            "error_info": error_info,
            "dialogue_length": compute_dialogue_length(messages)
        }
        conversations.append(conversation_entry)

    df = pd.DataFrame(conversations)
    print("Feedback Summary:")
    print(df['final_feedback'].dropna().unique())
    print(f"Successful conversations: {df['successful'].sum()} out of {len(df)}")
    return df


import numpy as np
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt', quiet=True)


def compute_median_dialogue_lengths(df, outlier_conversation_id=0):
    """
    Computes the median turn length per conversation, excluding the specified outlier.
    Returns a DataFrame with columns:
      'conversation_id' and 'median_turn_length'.

    :param df: DataFrame containing conversation data.
    :param outlier_conversation_id: The conversation_id to drop as outlier.
    :return: A DataFrame with 'conversation_id' and 'median_turn_length'.
    """
    # Exclude the outlier conversation (by ID)
    df_no_outlier = df[df["conversation_id"] != outlier_conversation_id].copy()

    if "turn_metrics" in df_no_outlier.columns:
        # Compute median of word counts per turn if turn_metrics/words_per_turn is available
        df_no_outlier["median_turn_length"] = df_no_outlier["turn_metrics"].apply(
            lambda metrics: int(np.median(metrics.get("words_per_turn", [])))
            if metrics and metrics.get("words_per_turn")
            else 0
        )
    else:
        # Fallback if "turn_metrics" does not exist.
        # For example, use 'dialogue_length' if it’s there, or default to 0.
        if "dialogue_length" in df_no_outlier.columns:
            df_no_outlier["median_turn_length"] = df_no_outlier["dialogue_length"]
        else:
            df_no_outlier["median_turn_length"] = 0

    # Build a small DataFrame with the relevant columns
    if "conversation_id" not in df_no_outlier.columns:
        # If conversation_id is missing, create a dummy ID or return an empty frame
        return pd.DataFrame(columns=["conversation_id", "median_turn_length"])

    result_df = df_no_outlier[["conversation_id", "median_turn_length"]].sort_values("conversation_id")
    return result_df


if __name__ == "__main__":
    # Run the pipeline only when executed directly.
    file_path = 'dataset_conversations.txt'
    df_conversations = process_conversations(file_path)

    output_csv = "processed_conversations.csv"
    df_conversations.to_csv(output_csv, index=False)
    print(f"Processed data saved to {output_csv}")
