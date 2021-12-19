import re

def clean_html(raw_html):
  clean_text = re.sub(re.compile('<.*?>'), '', raw_html)
  return clean_text