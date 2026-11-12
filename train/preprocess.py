import re

def preprocesses_text(text):
    '''This method will lower case all letters, replace special characters with space 
    and replace consecutive spaces or line breaks with a space. '''
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text