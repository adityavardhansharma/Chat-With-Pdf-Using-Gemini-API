import json
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import faiss
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=google_api_key)


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def save_vector_store(vector_store):
    index_config = {
        "index_type": type(vector_store.index).__name__,
        "embeddings_model": vector_store.embeddings.model,
    }
    with open("faiss_index.json", "w") as json_file:
        json.dump(index_config, json_file)


def load_vector_store():
    with open("faiss_index.json", "r") as json_file:
        index_config = json.load(json_file)
    index_type = index_config["index_type"]
    embeddings_model = index_config["embeddings_model"]

    dimension = 768  # Assuming the dimensionality of Google Generative AI embeddings is 768

    index_class = getattr(faiss, index_type)
    index = index_class(dimension)  # Pass the dimensionality to the IndexFlatL2 constructor

    embeddings = GoogleGenerativeAIEmbeddings(model=embeddings_model, google_api_key=google_api_key)

    # Create a FAISS object with the required arguments
    vector_store = FAISS(embeddings, index, docstore=None, index_to_docstore_id=None)
    return vector_store


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    save_vector_store(vector_store)
    return vector_store


def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-1.0-pro", temperature=1)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


def user_input(user_question):
    new_db = load_vector_store()
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])


def main():
    st.set_page_config("Chat With PDF")
    st.header("Chat with PDF using Gemini")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button",
                                    accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")


if __name__ == "__main__":
    main()
