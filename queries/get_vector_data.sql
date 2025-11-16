SELECT
    knowledge,
    (1 - (vector_knowledge <=> '@VECTOR_INPUT')) AS similarity
FROM lna_vector_data
ORDER BY similarity DESC
LIMIT 6