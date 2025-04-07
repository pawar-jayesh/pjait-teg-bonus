import os
import tempfile
import urllib.request
import json
import numpy as np

# from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import RetrievalQA, LLMChain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_community.retrievers import BM25Retriever
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document


# Load API keys from environment variables or .env file
load_dotenv()

# Configure OpenAI API access
openai_api_key = os.getenv("OPENAI_API_KEY_TEG")

# Configure our main language model - gpt-4o-mini
BASE_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-ada-002"

# Create LangChain model interface
llm = ChatOpenAI(api_key=openai_api_key, model=BASE_MODEL, temperature=0)
second_llm_eval = ChatOpenAI(api_key=openai_api_key, model=BASE_MODEL, temperature=0)
embeddings = OpenAIEmbeddings(api_key=openai_api_key, model=EMBEDDING_MODEL)

# Set up default configurations
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_RETRIEVER_K = 4

user_query = "What is the MedAgent-Pro?"


def download_sample_document(url=None):
    """Download a sample document from a URL or use a default document"""
    temp_dir = tempfile.mkdtemp()
    pdf_path = os.path.join(temp_dir, "sample-document.pdf")
    
    # Get paper url
    if url is None:
        url = "https://arxiv.org/pdf/2503.18968.pdf"
    
    # Download the document
    urllib.request.urlretrieve(url, pdf_path)
    print(f"Downloaded document to: {pdf_path}")
    return pdf_path, temp_dir

def load_and_split_document(file_path, chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP):
    """Load a document and split it into chunks"""
    # Determine loader based on file extension
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")
    
    # Load the document
    documents = loader.load()
    print(f"Loaded {len(documents)} document pages/segments")
    
    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    
    return chunks

def create_vectorstore(chunks, embedding_model=embeddings):
    """Create a vector store from document chunks"""
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    print(f"Created vector store with {len(chunks)} documents")
    return vectorstore


file_path, temp_dir = download_sample_document("https://arxiv.org/pdf/2503.18968.pdf")

# Load and split the document
chunks = load_and_split_document(file_path)

# Create the vector store
vectorstore = create_vectorstore(chunks)

# Create a basic retriever from the vectorstore we created in Part 1
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})


# RAG 1: Basic with Document Stuffing
def basic_rag(query):
    try:
        # """
        # Implement basic RAG with document stuffing
        # """
        # Create a standard RAG prompt
        prompt = PromptTemplate.from_template(
            """
            Answer the following question based only on the provided context:
            
            Context:
            {context}
            
            Question: {input}
            
            Answer:
            """
        )
        
        # Create a document chain that combines the documents
        document_chain = create_stuff_documents_chain(llm, prompt)
        
        # Create a retrieval chain that uses the retriever and document chain
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        # Run the chain
        result = retrieval_chain.invoke({"input": query})
        
        # Return the answer and the source documents
        return {
            "query": query,
            "answer": result["answer"],
            "source_documents": result["context"]
        }
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# print("-----------------------------------------------------------------------------------------------")
# print(basic_rag(user_query))




# RAG 2: Contextual compression RAG
def compression_rag(query):
    """
    Implement RAG with contextual compression to extract only the relevant parts of documents
    """
    # Create the document compressor using LLM to extract relevant information
    compressor = LLMChainExtractor.from_llm(llm)
    
    # Create a compressed retriever
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=retriever
    )
    
    # Create a standard RAG prompt
    prompt = PromptTemplate.from_template(
        """
        Answer the following question based only on the provided context:
        
        Context:
        {context}
        
        Question: {input}
        
        Answer:
        """
    )
    
    # Create a document chain that combines the compressed documents
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    # Create a retrieval chain with the compression retriever
    retrieval_chain = create_retrieval_chain(compression_retriever, document_chain)
    
    # Run the chain
    result = retrieval_chain.invoke({"input": query})
    
    # Return the answer and the source documents
    return {
        "query": query,
        "answer": result["answer"],
        "source_documents": result["context"]
    }

# print("-----------------------------------------------------------------------------------------------")
# print(compression_rag(user_query))





# RAG 3: Hybrid search RAG (Dense + Sparse retrieval)
def hybrid_rag(query):
    """
    Implement RAG with hybrid search (combining dense and sparse retrievers)
    """
    # Create a BM25 (sparse) retriever from the same documents
    docs_with_scores = vectorstore.similarity_search_with_score("", k=len(vectorstore.index_to_docstore_id))
    bm25_docs = [doc.page_content for doc, _ in docs_with_scores]
    
    
    bm25_retriever = BM25Retriever.from_texts(bm25_docs)
    bm25_retriever.k = 4
    
    # Create an ensemble retriever that combines dense and sparse
    ensemble_retriever = EnsembleRetriever(
        retrievers=[retriever, bm25_retriever],
        weights=[0.7, 0.3]  # Weight dense retrieval higher
    )
    
    # Create a standard RAG prompt
    prompt = PromptTemplate.from_template(
        """
        Answer the following question based only on the provided context:
        
        Context:
        {context}
        
        Question: {input}
        
        Answer:
        """
    )
    
    # Create a document chain
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    # Create a retrieval chain
    retrieval_chain = create_retrieval_chain(ensemble_retriever, document_chain)
    
    # Run the chain
    result = retrieval_chain.invoke({"input": query})
    
    # Return the answer and the source documents
    return {
        "query": query,
        "answer": result["answer"],
        "source_documents": result["context"]
    }


# print("-----------------------------------------------------------------------------------------------")
# print(hybrid_rag(user_query))




# Prompt 1 : Zero Shot
def zero_shot_prompt(task):
    """Basic zero-shot prompting"""
    response = llm.invoke(task)
    return response.content

# Test with a simple task
# zero_shot_task = "Explain how transformer neural networks work in 3 paragraphs."

# print("-----------------------------------------------------------------------------------------------")
# print("\nZero-Shot Prompting Example:\n")
# print(zero_shot_prompt(user_query))




# Prompt 2 : Chain of Thought
def chain_of_thought_prompt(task):
    """Chain-of-thought prompting for step-by-step reasoning"""
    prompt = f"{task}\n\nLet's think through this step by step:"
    response = llm.invoke(prompt)
    return response.content

# Test with a math problem
# math_problem = "If I buy 5 apples at $0.50 each and 3 oranges at $0.75 each, and I pay with a $10 bill, how much change do I get back?"
# print("-----------------------------------------------------------------------------------------------")
# print("\nChain-of-Thought Prompting Example:\n")
# print(chain_of_thought_prompt(user_query))




# Prompt 3 : Role Promting
def role_prompt(role, task):
    """Prompt the model to adopt a specific role or persona"""
    prompt = f"You are an expert {role}.\n\n{task}"
    response = llm.invoke(prompt)
    return response.content

# Test with a technical question
crypto_task = "Explain why quantum computing poses a risk to current encryption methods."
# role = "cryptographer with 20 years of experience in quantum-resistant algorithms"4
role = "an expert and prominent figure of the field related to the task"
# print("-----------------------------------------------------------------------------------------------")
# print("\nRole Prompting Example:\n")
# print(role_prompt(role, user_query))



# Function which evaluates RAG response
def evaluate_rag_response(query, answer, reference_answer=None):
    """
    Evaluate a RAG response based on various criteria
    """
    if reference_answer:
        evaluation_prompt = f"""
        Question: {query}
        
        Generated Answer: {answer}
        
        Reference Answer: {reference_answer}
        
        Evaluate this answer on a scale of 1-10 for:
        1. Relevance: How well the answer addresses the question
        2. Accuracy: How factually correct the information is compared to the reference
        3. Completeness: How thoroughly the answer covers the topic
        4. Conciseness: How focused and to-the-point the answer is
        
        Provide a score for each criterion and a brief explanation.
        """
    else:
        evaluation_prompt = f"""
        Question: {query}
        
        Generated Answer: {answer}
        
        Evaluate this answer on a scale of 1-10 for:
        1. Relevance: How well the answer addresses the question
        2. Accuracy: How factually correct the information appears to be
        3. Completeness: How thoroughly the answer covers the topic
        4. Conciseness: How focused and to-the-point the answer is
        
        Provide a score for each criterion and a brief explanation.
        """
    
    evaluation = second_llm_eval.invoke(evaluation_prompt).content
    return evaluation


# function which compares RAG architecture
def compare_rag_architectures(queries):
    """
    Compare different RAG architectures on a set of queries
    """
    architectures = {
        "Basic RAG": basic_rag,
        "Compression RAG": compression_rag,
        "Hybrid RAG": hybrid_rag,
    }
    
    results = {}
    for name, architecture_func in architectures.items():
        architecture_results = []
        print(f"\nRunning {name} architecture...")
        
        for query in queries:
            print(f"  Query: {query}")
            result = architecture_func(query)
            
            # Evaluate the response
            evaluation = evaluate_rag_response(query, result["answer"])
            
            architecture_results.append({
                "query": query,
                "answer": result["answer"],
                "evaluation": evaluation,
                "num_source_docs": len(result["source_documents"]) if "source_documents" in result else 0,
                "strategy": result.get("strategy", name)
            })
        
        results[name] = architecture_results
    
    # Generate a comparative analysis
    comparative_analysis = []
    for i, query in enumerate(queries):
        query_results = {}
        for name in architectures.keys():
            query_results[name] = results[name][i]
        
        analysis_prompt = """
        Compare the following RAG architecture responses for the query: "{query}"
        
        {responses}
        
        Which architecture provided the most useful response and why? Analyze differences in accuracy, 
        relevance, and completeness between the approaches.
        """.format(
            query=query,
            responses="".join([
            f"{name} Answer: {query_results[name]['answer']}\n\n"
            for name in architectures.keys()
            ])
        )
        
        comparative_analysis.append({
            "query": query,
            "comparison": second_llm_eval.invoke(analysis_prompt).content
        })
    
    return {
        "individual_results": results,
        "comparative_analysis": comparative_analysis
    }



# # Test the RAG architectures
# test_queries = [
#     "what is the MedAgent-Pro?",
#     "What are the key features of the MedAgent-Pro?",
#     "Perform a detailed comparison of MedAgent-Pro with Multi-modal Foundation Models"
# ]

# print("---------------------------------------------------------------------------------------------------")
# print("\nComparing RAG Architectures\n" + "="*50)
# comparison_results = compare_rag_architectures(test_queries)

# # Display comparative analysis
# print("\nComparative Analysis of RAG Architectures:\n")
# for analysis in comparison_results["comparative_analysis"]:
#     print(f"Query: {analysis['query']}")
#     print(f"Analysis: {analysis['comparison']}")
#     print("-"*80)



# Generate overall recommendation
recommendation_prompt = """
Based on the comparisons across all test queries:

1. Which RAG architecture generally performed the best?
2. What are the strengths and weaknesses of each approach?
3. When would you recommend each architecture for different use cases?

Provide a detailed recommendation for each architecture:
- Basic RAG
- Compression RAG
- Hybrid RAG
"""

# overall_recommendation = second_llm_eval.invoke(recommendation_prompt).content
# print("\nOverall Recommendation for RAG Architecture Selection:\n")
# print(overall_recommendation)

