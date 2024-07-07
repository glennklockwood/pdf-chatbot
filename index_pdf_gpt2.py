import os
import numpy
import fitz  # PyMuPDF
import faiss
import json
import transformers
import torch

TOKENIZER = transformers.GPT2Tokenizer.from_pretrained('gpt2')
MODEL = transformers.GPT2Model.from_pretrained('gpt2')
EMBEDDING_MODEL = MODEL

def extract_text_from_pdfs(pdf_directory):
    texts = []
    for filename in os.listdir(pdf_directory):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(pdf_directory, filename)
            with fitz.open(pdf_path) as doc:
                for page in doc:
                    print("Extracting text from", pdf_path, "page", page.number)
                    # texts.append(page.get_text())
                    # read all text, then chop it into lines of not more than N characters
                    text = page.get_text()
                    # strip all unicode and newlines from text
                    text = "".join([c if ord(c) < 128 else "" for c in text])
                    text = text.replace("\n", " ")

                    lines = [text[i:i+1500] for i in range(0, len(text), 500)]
                    texts.extend(lines)

    return texts

def get_embedding(text):
    """Gets embeddings from GPT-2
    """
    # Tokenize the input text
    inputs = TOKENIZER(text, return_tensors='pt')
    # Get the hidden states from the GPT-2 model
    with torch.no_grad():
        outputs = EMBEDDING_MODEL(**inputs)
    # Use the hidden state of the last token as the embedding
    return outputs.last_hidden_state[:, -1, :].squeeze().numpy()

def query_index(index, query_text, k=5):
    """Queries the FAISS index
    """
    query_embedding = get_embedding(query_text).reshape(1, -1)
    distances, indices = index.search(query_embedding, k)
    return distances, indices

def generate_faiss_index(texts, index_file='faiss_index'):
    # Load pre-trained model and tokenizer

    embeddings = numpy.array([get_embedding(doc) for doc in texts])
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Save the index to a file
    faiss.write_index(index, index_file)
    print(f"Saved FAISS index to {index_file}")

if __name__ == "__main__":

    if not os.path.exists("texts.json"):
        pdf_directory = "input_files"  # Update this path
        texts = extract_text_from_pdfs(pdf_directory)
        print(f"Extracted {len(texts)} texts from PDFs.")
        
        print("Saving texts to texts.json")
        with open("texts.json", "w") as f:
            json.dump(texts, f)
    else:
        print("Loading texts from texts.json")
        texts = json.load(open("texts.json", "r"))
        print(f"Loaded {len(texts)} texts from texts.json")

    # Generate the FAISS index
    print("Generating FAISS index")
    generate_faiss_index(texts)
