# LLM_Respose_Evaluation_Pipeline


This project contains a simple Python script to evaluate the reliability of AI
responses using retrieved context data.

The evaluation focuses on:
- Response relevance
- Hallucination / factual grounding
- Latency
- Cost estimation

--------------------------------------------------
LOCAL SETUP INSTRUCTIONS
--------------------------------------------------

1. Install Python 3.11

2. Create and activate a virtual environment

   python -m venv llmvenv

   Windows:
   llmvenv\Scripts\activate

3. Install dependencies

   pip install sentence-transformers scikit-learn numpy

4. Run the script

   python main.py

The script reads:
- A chat conversation JSON file
- A retrieved context vectors JSON file

It prints an evaluation report to the console.

--------------------------------------------------
PIPELINE ARCHITECTURE
--------------------------------------------------

1. Extract the latest user query and AI response from the chat JSON
2. Load the top retrieved context chunks from the vector database JSON
3. Convert text to embeddings
4. Measure relevance between user query and AI response
5. Split the AI response into factual claims
6. Check each claim against the retrieved context to detect hallucinations
7. Measure latency and estimate cost
8. Generate a final reliability score

--------------------------------------------------
WHY THIS APPROACH
--------------------------------------------------

This solution is intentionally simple.

- No additional LLM calls during evaluation
- No external APIs
- Deterministic and explainable results
- Easy to debug and review

The goal is to detect unsupported claims using retrieved evidence rather than
relying on another AI model to judge responses.

--------------------------------------------------
SCALABILITY AND COST CONTROL
--------------------------------------------------

The script is designed for real-time and large-scale usage.

- Uses local embedding models
- No network calls during evaluation
- Caches embeddings to avoid recomputation
- Limits evaluation to top retrieved context chunks
- Uses fast cosine similarity

These choices keep latency low and costs predictable, even when running the
evaluation on millions of AI responses per day.
