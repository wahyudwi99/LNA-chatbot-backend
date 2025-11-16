INSERT INTO lna_chatbot_log (
    input_prompt,
    result,
    input_token,
    output_token,
    retrieved_knowledge_base,
    embedding_similarity,
    created_at
) VALUES (%s, %s, %s, %s, %s, %s, %s)