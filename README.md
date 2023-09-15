# streamlit-neo4j-hackathon

## Import fewshot examples:

```
LOAD CSV WITH HEADERS FROM "https://raw.githubusercontent.com/tomasonjo/streamlit-neo4j-hackathon/main/fewshot.csv" AS row
MERGE (f:Fewshot {id:linenumber()})
SET f += row;

MATCH (f:Fewshot)
CALL apoc.ml.openai.embedding([f.Question], $openai_api_key) YIELD embedding
SET f.embedding = embedding;

CALL db.index.vector.createNodeIndex('fewshot', 'Fewshot', 'embedding', 1536, 'cosine');
```
