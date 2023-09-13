# Semantic Search with LLM Embeddings

Use the OpenAI embeddings api to perform semantic search on markdown files.

I wrote a blog post explaining semantic search and how this code was implemented. Read it here: [Implementing Semantic Search with OpenAI's Embeddings API](https://stemhash.com/semantic-search-with-llms/)

## Usage

Before running the code, make sure to obtain your [OpenAI API key](https://openai.com/product).
Then store your api key as an environment variable named "OPENAI_API_KEY".

Alternatively (though not recommended), modify line 20 in `semantic_search.py` by setting the value of `openai.api_key` to your api key. If you do this, keep the key in your local directory only and never share your key publicly.

### Running the solution

You can run the solution by executing the `semantic_search.py` script and passing the desired options. To see a list of available options, run:

```bash
python3 semantic_search.py --help
```
Make sure you have all required libraries installed. These are located in the `requirements.txt` file.

### Running the solution in a venv

Preferably, create a virtual environment (venv) to run the application. This will take care of installing all the required libraries.

1. On the terminal, navigate to the directory where you want to create your venv. Then execute:

   ```bash
   python -m venv <venv_name>
   ```
   where <venv_name> is the name you want to give the virtual environment

2. To activate the environment:

      On MacOS/Linx execute:

   ```bash
   source <venv_name>/bin/activate
   ```
      On Windows execute:

   ```bash
   .\<venv_name>\Scripts\activate
   ```
3. Now you can run the `semantic_search.py` script as before:

   ```bash
   python3 semantic_search.py --help
   ```
   to see the list of available options.

4. When you're done, deactivate the venv with:

   ```bash
   deactivate
   ```

## Notes

The processing of markdown files in this solution (`process_markdown.py` script) is tailored to a specific structure of the files. However, the main semantic search script (`semantic_search.py`) works with a generic list of dictionaries of the form:

```json
[
   {
      "Title": "<document title>",
      "Paragraph_id": "<paragraph number inside the document>",
      "Text": "<paragraph's text content>",
      "Embedding": "<text's embedding (initially empty)>"
   },
   {
      ...
   },
]
```
As long as this structure is followed as the input, the script will work.