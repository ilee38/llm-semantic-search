import numpy as np
import openai
import os
import pandas as pd
import pickle
import sys
import tiktoken
import time
from cli import parser
from process_markdown import process_markdown_folder
from tenacity import (retry, stop_after_attempt, wait_random_exponential)

# CONFIGS #
DF_FILENAME = "embeddings.csv"
QUERY_CACHE_FILENAME = "query_cache.pickle"
OPENAI_EMBEDDINGS_MODEL = "text-embedding-ada-002"
MAX_TOKENS = 8191 # Max number of tokens for OpenAI's Embeddings api
EMBEDDING_ENCODING = "cl100k_base"

openai.api_key = os.environ.get("OPENAI_API_KEY")

def get_num_tokens(text, encoding_name=EMBEDDING_ENCODING):
   """Returns the number of tokens in a string
   """
   encoding = tiktoken.get_encoding(encoding_name)
   num_tokens = len(encoding.encode(text))
   return num_tokens


def truncate_tokens(text, encoding_name=EMBEDDING_ENCODING, max_tokens=MAX_TOKENS):
   """Truncates the tokens in a string to have the max_tokens
   """
   encoding = tiktoken.get_encoding(encoding_name)
   return encoding.encode(text)[:max_tokens]


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_embedding_from_api(text):
   print(f"requesting embedding from api for {text}...")
   return np.array([0.454, 4.565, 0.645, -5.5657])
   """
   return await openai.Embedding.create(
      model=OPENAI_EMBEDDINGS_MODEL,
      input=text
      )["data"][0]["embedding"]
   """


def embed(data):
   """Obtains embeddings from OpenAI Embeddings Api
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


def generate_data_embeddings(folder_path):
   """Generates embeddings for text data and saves them to a csv data file.
   """
   texts = process_markdown_folder(folder_path)
   print("Done processing .MD files...")
   df = embed(texts)
   print("Done generating embeddings, saving df to file...")
   df.to_csv(DF_FILENAME)


def cosine_similarity(v1, v2):
   """Computes the cosine similarity between two vectors:
            cosine similarity = (A . B)/(||A||*||B||)
         where:
            A . B is the dot product of vectors A and B
            ||A|| is the L2 norm (euclidean norm) of vector A
            ||B|| is the L2 norm (euclidean norm) of vector B
   """
   dot_product = np.dot(v1, v2)
   norm_v1 = np.linalg.norm(v1)
   norm_v2 = np.linalg.norm(v2)
   return dot_product / (norm_v1 * norm_v2)


def search(query):
   """Performs semantic search with embeddings.
      Returns the search results in descending order of relevance, based on the
      cosine similarity distance between embeddings
   """
   # Load search index into dataframe
   try:
      df = pd.read_csv(DF_FILENAME, header=[0, 1])
      df.columns = pd.MultiIndex.from_tuples([tuple(["" if y.find("Unnamed") == 0 else y for y in x]) for x in df.columns])
      print(df)
   except FileNotFoundError as e:
      print(f"{e}. Run again with flag -p set to True to process files")
      sys.exit(1)

   # Load query cache if it exists. Otherwise create it.
   try:
      with open(QUERY_CACHE_FILENAME, 'rb') as f:
         query_cache = pickle.load(f)
   except OSError:
      query_cache = {}

   if query in query_cache:
      query_vector = query_cache[query]
   else:
      query_vector = get_embedding_from_api(query)
      query_cache[query] = query_vector
      with open(QUERY_CACHE_FILENAME, 'wb') as f:
         pickle.dump(query_cache, f)

   similarities = np.apply_along_axis(lambda x: cosine_similarity(x, query_vector), axis=0, arr=df)
   result = pd.Series(similarities, index=df.columns).sort_values(ascending=False)
   return result


def main():
   args = parser.parse_args()

   if args.preprocess == True:
      if args.folderpath is None:
         raise ValueError("Folder path not defined. Use flag -f with path to folder")
      generate_data_embeddings(args.folderpath)

   if args.query is not None:
      results = search(args.query)
      num_results = args.num if args.num <= len(results) else len(results)
      for i in range(num_results):
         print(f"{results[i]}\n")
   return


if __name__ == "__main__":
   main()
