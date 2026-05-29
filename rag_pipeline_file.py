# important inputs for the project
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv() # for importing the groq api from the environment 

# parser for fetching the output in string format
parser = StrOutputParser()

# loading the pdf
def loader(path):
    loader = PyPDFLoader(path)
    docs = loader.load()
    return docs

# splliting the text
def textsplliter(docs,chunk_size=500,chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)
    return chunks

# creating and storing the vectors
def create_vectorstore(chunks):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(
        documents = chunks,
        embedding= embedding_model,
    )
    return vector_store

# searching for similar documents
def similar_documents(query,vector_store):
    retriver = vector_store.as_retriever(search_kwargs={"k":5})
    results = retriver.invoke(query)
    
    content_list = [docs.page_content for docs in results]
    
    return content_list

# chat_history for mainatig the sensible conversation between user and llm 
chat_history = []

# function for taking the out =put form the model
def model_response(query,chat_history,vector_store):
    content_dictionary = similar_documents(query,vector_store)

    # joining the whole context into one text   
    context = "\n\n".join(content_dictionary)

    # model calling and the prompt making 
    prompt = ChatPromptTemplate.from_messages([
        ("system","you are the most intelactual and helpful assistant, answer based on this context {context}"),
        MessagesPlaceholder(variable_name = "chat_history"),
        ("human","{query}")]
    )

    chat_history.append(HumanMessage(query))

    model = ChatGroq(
        model="llama-3.1-8b-instant",
        )
    
    chain = prompt | model | parser

    result = chain.invoke({'chat_history':chat_history,'context':context,'query':HumanMessage(query)})

    # inserting the out into the chat_history
    chat_history.append(AIMessage(result))

    return result