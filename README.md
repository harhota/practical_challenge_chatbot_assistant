# AI Coaching Conversations Analysis

> **Note:** This project is intended as part of a test assignment for some company. Please consider the scope and purpose 
> of the project when reviewing.

This repository contains analysis of conversation data from an AI coaching platform. The project processes
raw conversation data, extracts both quantitative and qualitative metrics, and presents insights via an interactive 
Streamlit dashboard.

## Directory Structure

├── `app.py` – Streamlit app that imports functions from `main.py` (and `funcs.py`) to display the analysis.  
├── `funcs.py` – Contains data processing functions for cleaning and analyzing conversation data.   
├── `dataset_conversations.txt` – Raw conversation data in JSON/JSONL format.  
├── `conversations_research.ipynb` – Jupyter Notebook for raw data exploration and research.  
├── `requirements.txt` – List of dependencies for the project.  
└── `other output files...` – Processed files (e.g., CSV reports, `system_prompt.txt`, etc.).


> **Note:** The `conversations_research.ipynb`  notebook includes preliminary and 
> unrefined analysis. Reviewers should not expect finalized results in this notebook; it serves as a record of research 
> and exploratory work.

## Project Overview

The goal of this project is to extract actionable insights from a dataset of conversations generated by an AI coaching platform. Key features include:

- **Data Processing:** Parsing and cleaning raw conversation data, distinguishing JSONL from JSON array formats, and extracting key fields such as metadata, conversation turns, user feedback, and error information.
- **Defining Success:** Implementing heuristics (e.g., checking the last few messages for specific feedback keywords) to flag conversations as “successful.”
- **Metrics Calculation:** Computing dialogue lengths (word counts excluding system messages) and turn metrics (number of user vs. assistant turns and average turn length), with special handling for outliers.
- **Quantitative Analysis:** Presenting overall conversation metrics, such as total and successful conversation counts, and visualizing dialogue length distributions (with outliers excluded) and median lengths for individual dialogues.
- **Qualitative Analysis:** Exploring thematic insights from specific conversations (for example, conversation #3 with the highest average words per turn and conversation #17 with the lowest), with options to expand or hide detailed reports.
- **Conclusions**

## Setup and Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/your_project_directory.git
   cd your_project_directory
   
Install Dependencies:

Ensure you have Python 3.11 (or a compatible version) installed, then run:

`pip install -r requirements.txt`

### Launch the Streamlit app with:

`streamlit run app.py`

The dashboard in `app.py` provides main research insights 

**Research & Analysis**

For further insights and raw exploration of the data, please refer to the `conversations_research.ipynb` notebook. Though it is raw,
the notebook documents the initial experiments. Probably in the meantime it will be cleaned and re-uploaded. 
