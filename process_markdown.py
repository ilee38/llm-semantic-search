import markdown
import os
import re
from bs4 import BeautifulSoup

DATE_PATTERN = r"([A-Z][a-z]{2})\s+(\d{1,2}),\s+(\d{4})"

def extract_text_from_markdown(file_path):
   """Extracts the text from all .md files (located in 'file_path') as paragraph blocks:
      Returns a list of dictionaries with keys: 'Title', 'Paraph_id', 'Text' and 'Embedding'
      The Embedding value is left empty as it will be filled when processing all embeddings.
   """
   document_paragraphs = []
   paragraph_num = 0

   with open(file_path, 'r', encoding='utf-8') as file:
      markdown_text = file.read()

   # Convert Markdown to HTML
   html = markdown.markdown(markdown_text)

   # Parse HTML and extract text
   soup = BeautifulSoup(html, 'html.parser')
   title = soup.find('h1')
   paragraphs = soup.find_all('p')

   for p in paragraphs:
      if p.get_text() == "" or p.get_text() is None:
         continue
      elif re.match(DATE_PATTERN, p.get_text()) is not None:
         continue
      # Check if paragraph is just new lines and skip it
      text = p.get_text().replace("\n", "")
      if text == "":
         continue

      p = f"{p.get_text()}".rstrip()
      paragraph_num += 1
      document_paragraphs.append({
         "Title": title.get_text(),
         "Paragraph_id": str(paragraph_num),
         "Text": p,
         "Embedding": None
      })

   return document_paragraphs


def process_markdown_folder(folder_path):
   """Processes all .md files in the folder.
      Returns an array of dictionaries with the text of all files separated by paragraph as:
      ]
         {
            "Title": "<title>",
            "Paragraph_id": "<paragraph number>",
            "Text": "<text in paragraph>",
            "Embedding": None
         }
      ]
   """
   all_texts = []
   if not os.path.exists(folder_path):
      print(f"Folder '{folder_path}' does not exist.")
      return

   for filename in os.listdir(folder_path):
      if filename.endswith('.md'):
         file_path = os.path.join(folder_path, filename)
         file_sections = extract_text_from_markdown(file_path)
         for i in range(len(file_sections)):
            all_texts.append(file_sections[i])

   return all_texts
