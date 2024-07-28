import os
import numpy as np
import torch
from torch.nn import CosineSimilarity
import transformers
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

HF_TOKEN = os.getenv("HF_TOKEN")
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.device_count() > 0 else "cpu"
MAX_ANSWER_LENGTH = 400
CHAT_MODEL_PATH = "models/meta-llama-model"
TOKENIZER_PATH = "models/meta-llama-tokenizer"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
FILES = os.listdir("data/raw")
MODE = "text-generation"
NUM_DOCUMENTS_RETRIEVED = 5
SYSTEM_PROMPT_TEMPLATE = """
Du bist Osrambot, ein KI-Assistent, welcher Kundenanfragen zu Produkten von OSRAM beantworten soll. 

Hier sind einige Dokumente die für die letzten Anfragen des Kunden relevant sind:

{relevant_documents}
"""

# use recursive splitting as a default
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)

model = transformers.AutoModelForCausalLM.from_pretrained(CHAT_MODEL_PATH, torch_dtype=torch.bfloat16)
tokenizer = transformers.AutoTokenizer.from_pretrained(TOKENIZER_PATH)
pipeline = transformers.pipeline(
    MODE,
    model=model,
    tokenizer=tokenizer,
    model_kwargs={"torch_dtype": torch.bfloat16},
    token=HF_TOKEN,
    device=DEVICE,
    pad_token_id=tokenizer.eos_token_id
)

embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

all_pages = []
for file in FILES:
    loader = PyPDFLoader("data/raw/" + file)
    pages = loader.load_and_split()
    all_pages.extend(pages)
#print(f"Number of documents: {len(all_pages)}")
chunked_documents = text_splitter.split_documents(all_pages)
#print(f"Number of document chunks: {len(chunked_documents)}")
chunked_documents_page_content = [doc.page_content for doc in chunked_documents]
#print("First document:")
#print(chunked_documents[0])
# add the document embeddings to the document chunks
for doc in chunked_documents:
    doc.metadata["embedding"] = embedding_model.embed_query(doc.page_content)
# print("First document embedding:")
# print(chunked_documents_embedding[0])
#print("Embedding size:")
#print(len(chunked_documents[0].metadata["embedding"]))

# create in-memory vector store since there are not that many documents
vector_store = [doc.metadata["embedding"] for doc in chunked_documents]
vector_store = torch.tensor(vector_store)

# use cosine similarity to compute the most relevant document embeddings
cos = CosineSimilarity(dim=1)

new_chat = True
session_query = ""
session_history = []
print("New session ------------------------------------------------------------------------------------------------------------")
while True:
    if new_chat:
        print("Type your question related to Osram products below and confirm with <Enter>. To restart your chat session, enter '/new_chat'.")
        print("Willkommen bei Osrambot, wie kann ich Ihnen helfen?")
        new_chat = False
    user_utterance = input("User: ")
    print("------")
    session_history.append({
        "role": "user",
        "content": user_utterance
    })
    if user_utterance == "/new_chat":
        new_chat = True
        session_query = ""
        session_history = []
        print("New session ------------------------------------------------------------------------------------------------------------")
        continue
    # use only user utterances to prevent the answer from guiding the retrieval step
    session_query += user_utterance + " "
    session_query_embedding = torch.tensor(embedding_model.embed_query(session_query))
    similarities = cos(session_query_embedding, vector_store)
    # get top K documents
    _, top_k_indices = torch.topk(similarities, NUM_DOCUMENTS_RETRIEVED, sorted=True)
    relevant_documents_list = [chunked_documents[i] for i in top_k_indices.tolist()]
    # concatenate the relevant to a string to pass them as the system prompt
    relevant_documents = "---".join([doc.page_content for doc in relevant_documents_list])
    conversation_turn_prompt = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT_TEMPLATE.format(
                    relevant_documents=relevant_documents
                )
            }
        ]
    conversation_turn_prompt.extend(session_history)
    assistant_utterance = pipeline(conversation_turn_prompt, 
        max_new_tokens=MAX_ANSWER_LENGTH
    )
    # print the assistant utterance and the most relevant document
    print(f"O-Bot: {assistant_utterance[0]['generated_text'][-1]['content']}")
    print(f"Most relevant document: {relevant_documents_list[0].metadata['source']}, page {relevant_documents_list[0].metadata['page']}")
    print("------")
    session_history.append(assistant_utterance[0]['generated_text'][-1])
