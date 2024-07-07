import json
import transformers
import faiss
from index_pdf_gpt2 import get_embedding, TOKENIZER

MODEL = transformers.GPT2LMHeadModel.from_pretrained('gpt2')

def generate_response(question, context):

    input_text = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
    print(f"=== Start of input text===\n{input_text}\n=== End of input text===")

    input_len = len(input_text)
    input_len = 0

    generator = transformers.pipeline(
        "text-generation",
        model=MODEL,
        tokenizer=TOKENIZER
    )

    # see https://huggingface.co/blog/how-to-generate
    outputs = generator(
        input_text,
        max_new_tokens=50,
        num_beams=5,
        do_sample=True,
        early_stopping=True,
    )

    print("Output:\n" + 80 * '-')
    for i, output in enumerate(outputs):
        print("{}: {}".format(
              i,
              output['generated_text'][input_len:]))

    answer = outputs[0]['generated_text'][input_len:]
    return answer

# Load the FAISS index and texts
vectorstore = faiss.read_index("faiss_index")
with open("texts.json", "r") as f:
	texts = json.load(f)

def main(question):
    # Encode the question and search in the FAISS index using the SentenceTransformer model
    #question_vector = sentence_model.encode([question])
    question_vector = get_embedding(question).reshape(1, -1)

    D, I = vectorstore.search(question_vector, k=2)
    print(f"FAISS Search Indices: {I}")
    print(f"FAISS Search Distances: {D}")
    print(f"Using contexts: {I[0]}")

    context = " ".join([texts[i] for i in I[0]])
    print(f"Context is {len(context.split())} words long.")
	
	# Generate response using GPT-2
    answer = generate_response(question, context)
    print(f"=== Begin answer ===\n{answer}\n=== End answer ===")

if __name__ == "__main__":
	main("Based on the context, how many bids were submitted?")