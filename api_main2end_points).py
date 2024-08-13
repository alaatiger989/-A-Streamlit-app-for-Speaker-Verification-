import time
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
import nemo.collections.asr as nemo_asr
from pydantic import BaseModel
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List
import ast
import json
app = FastAPI()


@st.cache_resource
def load_model():
    return nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name='titanet_large')


speaker_model = load_model()

class SimilarityResponse(BaseModel):
    similarity: str
    decision: str
class EmbeddingsResponse(BaseModel):
    embeddings: list

# def compute_similarity(embeddings1, embeddings2):
#     similarity = cosine_similarity([embeddings1], [embeddings2])[0]
#     return similarity
def compute_similarity(embeddings1, embeddings2):
    # Flatten the embeddings if they are not already 1-dimensional
    embeddings1 = np.array(embeddings1).flatten()
    embeddings2 = np.array(embeddings2).flatten()

    # Print shapes for debugging
    print(f"Shape of embeddings1: {embeddings1.shape}")
    print(f"Shape of embeddings2: {embeddings2.shape}")

    similarity = cosine_similarity([embeddings1], [embeddings2])[0][0]
    return similarity
def extract_embeddings(file_path):
    start_time = time.time()
    embs = speaker_model.get_embedding(file_path)
    end_time = time.time() - start_time
    return embs.cpu().numpy(), end_time


@app.post("/extract_embeddings", response_model=EmbeddingsResponse)
async def extract_embeddings_endpoint(file: UploadFile = File(...)):
    file_location = f"temp_{file.filename}"
    print(file_location)
    with open(file_location, "wb") as f:
        f.write(await file.read())

    embs, processing_time = extract_embeddings(file_location)
    print(f"Processing time: {processing_time} seconds")

    return {"embeddings": embs.tolist()}

@app.post("/verify_similarity", response_model=SimilarityResponse)
async def verify_similarity(file: UploadFile = File(...), embeddings: str = Form(...)):
    #print(f"Received embeddings: {embeddings}")  # Debug: Print received embeddings
    if embeddings is None:
        raise HTTPException(status_code=400, detail="Embeddings vector is required")

    embeddings = json.loads(embeddings)
    embeddings_list = ast.literal_eval(embeddings)
    processed_received_embeddings = np.array(embeddings_list, dtype=np.float32)
    print("\nProcessed Embeddings : \n" , processed_received_embeddings)
    print("\n" ,processed_received_embeddings.dtype)


    file_location = f"temp_{file.filename}"
    print(file_location)

    with open(file_location, "wb") as f:
        f.write(await file.read())

    generated_embs, processing_time = extract_embeddings(file_location)
    print(f"Processing time: {processing_time} seconds")
    print("\nExtracted Embeddings: \n ", generated_embs , "\n" , generated_embs.dtype)
    similarity = compute_similarity(processed_received_embeddings, generated_embs)
    # Convert similarity to percentage
    similarity_percentage = similarity * 100
    # Clamp the similarity percentage to be within 0 to 100
    if similarity_percentage < 0:
        similarity_percentage = 0
    elif similarity_percentage > 100:
        similarity_percentage = 100
    # Make the decision based on the percentage
    decision = "verified" if similarity_percentage > 45 else "not verified"
    #decision = "verified" if similarity > 0.45 else "not verified"  # Threshold can be adjusted
    similarity_percentage = f"{similarity_percentage:.2f}%"
    return {"similarity": str(similarity_percentage), "decision": decision}


# @app.post("/verify_similarity", response_model=SimilarityResponse)
# async def verify_similarity(file: UploadFile = File(...), embeddings: str = Form(...)):
#     #print(f"Received embeddings: {embeddings}")  # Debug: Print received embeddings
#     if embeddings is None:
#         raise HTTPException(status_code=400, detail="Embeddings vector is required")
#
#     embeddings = json.loads(embeddings)
#     embeddings_list = ast.literal_eval(embeddings)
#     processed_received_embeddings = np.array(embeddings_list, dtype=np.float32)
#     print("\nProcessed Embeddings : \n" , processed_received_embeddings)
#     print("\n" ,processed_received_embeddings.dtype)
#     # # Parse the embeddings vector from a JSON string
#     # try:
#     #
#     #
#     #     if not isinstance(embeddings, list) or not all(isinstance(x, float) for x in embeddings):
#     #         raise ValueError
#     # except (json.JSONDecodeError, ValueError):
#     #     raise HTTPException(status_code=400, detail="Invalid embeddings vector format")
#
#     file_location = f"temp_{file.filename}"
#     print(file_location)
#
#     with open(file_location, "wb") as f:
#         f.write(await file.read())
#
#     generated_embs, processing_time = extract_embeddings(file_location)
#     print(f"Processing time: {processing_time} seconds")
#     print("\nExtracted Embeddings: \n " , generated_embs , "\n" , generated_embs.dtype)
#     #similarity = compute_similarity(processed_received_embeddings, generated_embs)
#     similarities = np.dot(processed_received_embeddings, generated_embs.T)
#     max_similarity = similarities.max()
#     print(max_similarity)
#     decision = "verified" if max_similarity > 0.75 else "not verified"  # Threshold can be adjusted
#
#     return {"similarity": max_similarity, "decision": decision}
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
