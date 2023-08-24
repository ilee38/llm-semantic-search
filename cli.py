import argparse

parser = argparse.ArgumentParser(
    prog="semantic_search",
    description="Semantic search with LLM Embeddings"
)

parser.add_argument(
    "-p",
    "--preprocess",
    default=False,
    help="If set to True, makes the script process the markdown files and generate embeddings. Requires the folder path passed using the -f flag.",
    required=False,
    type=bool
)
parser.add_argument(
    "-f",
    "--folderpath",
    help="Full path to the folder where the text (.md) files are contained.",
    required=False,
    type=str
)
parser.add_argument(
    "-q",
    "--query",
    help="The text to query for in the search index.",
    required=False,
    type=str
)
parser.add_argument(
    "-n",
    "--num",
    default=10,
    help="Number of results to return. If not specified, returns 10 results.",
    required=False,
    type=int
)