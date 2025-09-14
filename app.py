# from flask import Flask, render_template, request
# from src.helper import download_hugging_face_embeddings
# from langchain_community.vectorstores import Chroma
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_openai import OpenAI
# from langchain_core.prompts import ChatPromptTemplate
# from dotenv import load_dotenv
# from src.prompt import system_prompt
# import os

# # Setup Flask
# app = Flask(__name__)

# # Load environment variables
# load_dotenv()
# OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# # Load embeddings
# embeddings = download_hugging_face_embeddings()

# # Load ChromaDB Vector Store (persistent)
# db = Chroma(
#     persist_directory="chroma_db/",
#     embedding_function=embeddings
# )

# db.persist()

# # Build retriever and RAG chain
# retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
# llm = OpenAI(temperature=0.4, max_tokens=500)

# prompt = ChatPromptTemplate.from_messages([
#     ("system", system_prompt),
#     ("human", "{input}")
# ])

# question_answer_chain = create_stuff_documents_chain(llm, prompt)
# rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# # Routes
# @app.route("/")
# def index():
#     return render_template("index.html")  

# @app.route("/get", methods=["POST"])
# def chat():
#     msg = request.form["msg"]
#     response = rag_chain.invoke({"input": msg})
#     return str(response["answer"])

# # Run the app
# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=8080, debug=True)



from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
from langchain_chroma import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import OpenAI
from dotenv import load_dotenv
import os

# Setup Flask
app = Flask(__name__)

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Load embeddings
embeddings = download_hugging_face_embeddings()

# Load ChromaDB Vector Store (persistent)
db = Chroma(
    persist_directory="chroma_db/",
    embedding_function=embeddings
)

# Setup retriever
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Setup LLM
llm = OpenAI(temperature=0.4, max_tokens=500)

# Setup memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Setup Conversational RAG Chain
rag_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=False
)

# Routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]
    response = rag_chain.invoke({"question": msg})
    return str(response["answer"])

# Run the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
