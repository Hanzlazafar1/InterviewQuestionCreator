import streamlit as st
import os
import tempfile
from langchain.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain import HuggingFaceHub, PromptTemplate, LLMChain
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from transformers import GPT2TokenizerFast

# Set up environment variables
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
os.environ["GROQ_API_KEY"] = "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"
os.environ["GOOGLE_API_KEY"] = "AIzaSyAJh8E-TWlBX_3cUjNyS9gCNBbgCrrBNSs"

# Load the GPT2 tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

# Define the question generation process
def generate_questions_from_pdf(pdf_file):
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(pdf_file.getbuffer())
        temp_file_path = temp_file.name
    
    # Load PDF file
    loader = PyPDFLoader(temp_file_path)
    data = loader.load()
    
    # Combine page content into one string
    question_gen = ""
    for page in data:
        question_gen += page.page_content

    # Split the content into manageable chunks
    text_splitter = CharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer, chunk_size=100, chunk_overlap=50
    )
    texts = text_splitter.split_text(question_gen)
    doc_ques_gen = [Document(page_content=t) for t in texts]

    # Use the Groq model for question generation
    llm = ChatGroq(
        model_name="llama3-groq-8b-8192-tool-use-preview",
        temperature=0.3
    )

    prompt = PromptTemplate(
        input_variables=['texts'],
        template="""You are an expert at creating questions based on coding materials and documentation.
        Your goal is to prepare a coder or programmer for their exam and coding tests.
        You do this by asking questions about the text below:

        {texts}

        Create questions that will prepare the coders or programmers for their tasks.
        Make sure not to lose any important information.

        Questions:
        """
    )

    simple_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=True
    )

    ques = simple_chain.run(texts)
    return ques.split("\n")

# Define the answer generation process
def generate_answers(questions, doc_ques_gen):
    # Generate embeddings for the documents
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_documents(doc_ques_gen, embeddings)

    # Set up the answer generation chain
    llm_answer_gen = ChatGroq(temperature=0.1, model="Llama3-8b-8192")
    answer_generation_chain = RetrievalQA.from_chain_type(
        llm=llm_answer_gen,
        chain_type="stuff",
        retriever=vector_store.as_retriever()
    )

    # Generate answers for each question
    answers = []
    for question in questions:
        if question.strip():
            answer = answer_generation_chain.run(question)
            answers.append((question, answer))
    return answers

# Streamlit app layout
def main():
    st.title("Interview Question and Answer Generator")
    st.write("Upload a PDF to generate interview questions and answers for programming tasks.")

    # Upload PDF file
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file is not None:
        st.write("Processing the PDF file...")

        # Generate questions from the uploaded PDF
        questions = generate_questions_from_pdf(uploaded_file)
        
        # Display the generated questions
        st.subheader("Generated Questions")
        for idx, question in enumerate(questions):
            st.write(f" {question}")

        # Generate answers for the questions
        if st.button("Generate Answers"):
            st.write("Generating answers for the questions...")
            doc_ques_gen = [Document(page_content=txt) for txt in questions]
            answers = generate_answers(questions, doc_ques_gen)
            
            # Display the answers
            st.subheader("Generated Answers")
            for idx, (question, answer) in enumerate(answers):
                st.write(f"**Question {idx + 1}:** {question}")
                st.write(f"**Answer:** {answer}")
                st.write("------")

if __name__ == "__main__":
    main()
