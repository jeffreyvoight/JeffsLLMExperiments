# import json
# from os import listdir
# from os.path import isfile, join
import re
# from dotenv import load_dotenv
from pathlib import Path

class Text:
    def __init__(self, filename):
        self.title = re.match(r".+\\(.+)\.txt$", filename).group(1)
        self.author = re.match(r".+\\(\w+)\\.+.txt", filename).group(1)
        self.filename = f"{self.author}-{self.title}"
        self.metadata = {
            "author": self.author,
            "title": self.title,
            "filename": self.filename
            }
        with open(filename, encoding="utf-8-sig") as f:
            file_contents = f.read()
            self.contents = re.sub(r'[^\S\r\n]+', " ", file_contents)
        

def load_directory(dirname, filter):
    return list(Path(dirname).rglob(filter))
