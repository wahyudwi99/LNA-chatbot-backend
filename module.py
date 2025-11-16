import os
import pytz
import psycopg2
import warnings
import pandas as pd
from google import genai
from datetime import datetime
import torch.nn.functional as F
from torch import Tensor
from template_prompt import template_prompt_chatbot

QUERY_DIR_PATH="./queries"

def postgresql_connect():
    """
    Connect to postgresql database
    """
    database_client = psycopg2.connect(
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USERNAME"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT")
    )

    return database_client


def insert_data_to_db(data: dict):
    """
    Insert processed data to database
    """
    connection = postgresql_connect()
    cursor = connection.cursor()

    with open(f"{QUERY_DIR_PATH}/insert_data.sql", "r") as openfile:
        query_file = openfile.read()

    values = (
        data["input_prompt"],
        data["result"],
        data["input_token"],
        data["output_token"],
        data["retrieved_knowledge_base"],
        round(float(data["embedding_similarity"]), 2),
        datetime.now(pytz.timezone("Asia/Jakarta")).strftime("%Y-%m-%d %H:%M:%S")
    )
    cursor.execute(query_file, values)
    connection.commit()

    cursor.close()
    connection.close()


def average_pooling(last_hidden_states: Tensor,
                    attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)

    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def retrieve_relevant_data(input_prompt: str,
                           embedding_model,
                           embedding_tokenizer):
    warnings.filterwarnings("ignore")

    # Vectorization (embedding)
    reconstructed_input_prompt = f"query: {input_prompt}"
    input_token_data = embedding_tokenizer(reconstructed_input_prompt, max_length=512, padding=True, truncation=True, return_tensors="pt")
    embedding_output = embedding_model(**input_token_data)
    embedding_output = average_pooling(embedding_output.last_hidden_state, input_token_data["attention_mask"])
    normalized_embedding_output = F.normalize(embedding_output, p=2, dim=1)
    vector_data = normalized_embedding_output.detach().numpy().tolist()

    # Get vector data
    connection = postgresql_connect()
    with open(f"{QUERY_DIR_PATH}/get_vector_data.sql", "r") as openfile:
        query = openfile.read()
        query = query.replace('@VECTOR_INPUT', str(vector_data[0]))
    
    df = pd.read_sql_query(query, connection)
    connection.close()

    # Select best matched data
    context = df["knowledge"].values[0]
    similarity = df["similarity"].values[0]
    
    return context, similarity


def construct_response(input_prompt: str,
                       embedding_model,
                       embedding_tokenizer):
    """
    Construct LLM response for every input prompt from users
    """
    # Define the google client service
    client = genai.Client(api_key=os.getenv("GOOGLE_CLOUD_API_KEY"))

    # Retrieve relevant context
    context, embedding_similarity = retrieve_relevant_data(
        input_prompt,
        embedding_model,
        embedding_tokenizer
    )

    template_prompt = template_prompt_chatbot.replace('@CONTEXT', context)

    # Get response from llm
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=[
            template_prompt,
            input_prompt
        ]
    )

    result = response.text
    response_data = {
        "input_prompt": input_prompt,
        "result": result,
        "input_token": response.usage_metadata.prompt_token_count,
        "output_token": response.usage_metadata.candidates_token_count,
        "retrieved_knowledge_base": context,
        "embedding_similarity": embedding_similarity
    }

    # Insert response data to database
    insert_data_to_db(response_data)

    return result