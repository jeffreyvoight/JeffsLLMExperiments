from ollama import Client
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re
import json
import os.path
from os import listdir
from os.path import isfile, join

SYSTEM_PROMPT = """
You are a helpful reading assistant who answers questions
based on snippets of text provided in context. Answer only using the context provided,
being as concise as possible. If you're unsure, just say that you don't know.
Context:
"""

def parse_file(filename, path, chunking_size):
    paragraphs = []
    fullpath = path + "/" + filename
    # print(fullpath)
    with open(path + "/" + filename, encoding="utf-8-sig") as f:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunking_size,
            chunk_overlap=chunking_size / 10,
            length_function=len,
            is_separator_regex=False
        )
        file_contents = f.read()
        file_contents = re.sub(r'[^\S\r\n]+', " ", file_contents)
        texts=text_splitter.create_documents([file_contents])
        # print(texts[0])
    for text in texts:
        paragraphs.append(text.page_content.strip())

    return paragraphs

def parse_directory(dirname):
    paragraphs = []
    onlyfiles = [f for f in listdir(dirname) if isfile(join(dirname, f))]
    # print(onlyfiles)
    for f in onlyfiles:
        paragraphs.append(parse_file(f, dirname))
    return paragraphs

def save_embeddings(filename, embeddings):
    if not os.path.exists("embeddings"):
        os.makedirs("embeddings")
    with open(f"embeddings/{filename}.json", "w") as f:
        json.dump(embeddings, f)

def load_embeddings(filename):
    if not os.path.exists(f"embeddings/{filename}.json"):
        return False
    with open(f"embeddings/{filename}.json", "r") as f:
        return json.load(f)

def get_directory_embeddings(dirname, client, modelname, chunks):
    if (embeddings := load_embeddings(dirname)) is not False:
        return embeddings
    
    # print(chunks[0])
    
    embeddings = [
        client.embeddings(model=modelname, prompt=chunk[0])["embedding"]
        for chunk in chunks
    ]
    save_embeddings(dirname, embeddings)
    return embeddings

def get_embeddings(filename, client, modelname, chunks):
    if (embeddings := load_embeddings(filename)) is not False:
        return embeddings
    
    embeddings = [
        client.embeddings(model=modelname, prompt=chunk)["embedding"]
        for chunk in chunks
    ]
    save_embeddings(filename, embeddings)
    return embeddings

def find_most_similar(needle, haystack):
    needle_norm = np.norm(needle)
    similarity_scores = [
        np.dot(needle, item) / (needle_norm * np.norm(item)) for item in haystack
    ]
    return sorted(zip(similarity_scores, range(len(haystack))), reverse=True)

def get_prompt_embeddings(client, embed_model, prompt):
    return client.embeddings(model=embed_model, prompt=prompt)[
        "embedding"
    ]

def generate_response(prompt, paragraphs, embeddings, client, gen_model, embed_model):
    
    most_similar_chunks = find_most_similar(get_prompt_embeddings(client, embed_model, prompt), embeddings)[:chunk_count]
    # for item in most_similar_chunks:
    #     print(item[0], paragraphs[item[1]])
    print("\n\n\n")
    system_prompt = SYSTEM_PROMPT + "\n".join(paragraphs[item[1]][0] for item in most_similar_chunks)
    # print(system_prompt)
    response = client.chat(
        model,
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        stream = False
    )
    return response