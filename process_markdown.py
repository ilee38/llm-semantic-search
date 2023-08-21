import os
import markdown
from bs4 import BeautifulSoup
import re

DATE_PATTERN = r"([A-Z][a-z]{2})\s+(\d{1,2}),\s+(\d{4})"

def extract_text_from_markdown(file_path):
   """ Extracts the text from the .md file as paragraph blocks:
      Returns a dictionary with the paragraphs text as {(doc title, paragraph number): paragraph text}
   """
   document_paragraphs = {}
   paragraph_num = 0

   with open(file_path, 'r', encoding='utf-8') as file:
      markdown_text = file.read()

   # Convert Markdown to HTML
   html = markdown.markdown(markdown_text)

   # Parse HTML and extract text
   soup = BeautifulSoup(html, 'html.parser')
   title = soup.find('h1')
   paragraphs = soup.find_all('p')

   # Append the title of the document to each paragraph for added context
   for p in paragraphs:
      if p.get_text() == "" or p.get_text() is None:
         continue
      elif re.match(DATE_PATTERN, p.get_text()) is not None:
         continue
      # Check if paragraph is just new lines and skip it
      text = p.get_text().replace("\n", "")
      if text == "":
         continue

      p = f"{title.get_text()}. {p.get_text()}"
      paragraph_num += 1
      document_paragraphs[(title.get_text(), str(paragraph_num))] = p

   return document_paragraphs

def process_markdown_folder(folder_path):
   """ Processes all .md files in the folder.
      Returns a dictionary with the text of all files separated by pargraph as:
      {
         ("title", "paragraph number"): "paragraph text",
         ...
      }
   """
   all_texts = {}
   if not os.path.exists(folder_path):
      print(f"Folder '{folder_path}' does not exist.")
      return

   for filename in os.listdir(folder_path):
      if filename.endswith('.md'):
         file_path = os.path.join(folder_path, filename)
         file_sections = extract_text_from_markdown(file_path)
         for (title, section_id), section_content in file_sections.items():
            all_texts[(title, section_id)] = section_content

   return all_texts
