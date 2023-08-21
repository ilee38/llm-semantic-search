"""
   General flow:
   1. Data pre-processing: take blog posts and chunk them
   2. create vectors with each post chunk (include the blog post's title to add context to the chunks)
   3. Save vectors to DB
   4. Create query and generate embedding for the query
   5. Search using cosine similarity
"""
import numpy as np
import openai
import os
import pandas as pd
from process_markdown import process_markdown_folder
import tiktoken

# CONFIGS #
FOLDER_PATH = ""
DF_FILENAME = "LLM-SemanticSearch/embeddings.csv"
QUERY_CACHE_FILENAME = "LLM-SemanticSearch/query_cache.csv"
OPENAI_EMBEDDINGS_MODEL = "text-embedding-ada-002"
MAX_TOKENS = 8191
EMBEDDING_ENCODING = "cl100k_base"

openai.api_key = os.environ.get("OPENAI_API_KEY")

def get_num_tokens(text, encoding_name=EMBEDDING_ENCODING):
   """ Returns the number of tokens in a string
   """
   encoding = tiktoken.get_encoding(encoding_name)
   num_tokens = len(encoding.encode(text))
   return num_tokens

def truncate_tokens(text, encoding_name=EMBEDDING_ENCODING, max_tokens=MAX_TOKENS):
   """ Truncates the tokens in a string to have the max_tokens
   """
   encoding = tiktoken.get_encoding(encoding_name)
   return encoding.encode(text)[:max_tokens]

async def get_embedding_from_api(text):
   return await openai.Embedding.create(
      model=OPENAI_EMBEDDINGS_MODEL,
      input=text
      )["data"][0]["embedding"]

def embed(data):
   """ Obtains embeddings from OpenAI Embeddings Api
      Returns a data frame with the generated embeddings
   """
   responses = {}

   for (title, id), text in data.items():
      num_tokens = get_num_tokens(text)
      if num_tokens > MAX_TOKENS:
         text = truncate_tokens(text)
      try:
         embedding = get_embedding_from_api(text)
         responses[(title, id)] = embedding
      except Exception as e:
         print(f"Error fetching embedding for {title} {id} from api: ", e)
         continue

   df = pd.DataFrame(responses)
   return df

def generate_embeddings():
   """ Generates embeddings for text data and saves them to a csv data file.
   """
   texts = process_markdown_folder(FOLDER_PATH)
   print("Done processing .MD files...")
   df = embed(texts)
   print("Done generating embeddings, saving df to file...")
   df.to_csv(DF_FILENAME)

if __name__ == "__main__":
   generate_embeddings()