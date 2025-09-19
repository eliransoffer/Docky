# main.py - Simplified RAG Pipeline
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate


# Load environment
os.chdir('C:/Users/eliranso/Desktop/portfolio/Docky')
load_dotenv()

class SimpleRAG:
	def __init__(self, pdf_path: str, persist_dir: str = "./chroma_db"):
		self.pdf_path = pdf_path
		self.persist_dir = persist_dir
		self.embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
		self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
		self.vector_store = None
		self.rag_chain = None

	def load_and_process_documents(self):
		"""Load PDF and create vector store"""
		# Load documents
		loader = PyPDFLoader(self.pdf_path)
		docs = loader.load()

		# Split documents
		splitter = RecursiveCharacterTextSplitter(
			chunk_size=1000,
			chunk_overlap=200,
			add_start_index=True
		)
		splits = splitter.split_documents(docs)

		# Create or load vector store
		self.vector_store = Chroma(
			collection_name="rag_collection",
			embedding_function=self.embeddings,
			persist_directory=self.persist_dir
		)

		# Add documents if database is empty
		if len(self.vector_store.get()["ids"]) == 0:
			self.vector_store.add_documents(splits)

		return len(splits)

	def setup_chain(self):
		"""Setup the RAG chain"""
		# Simple prompt template
		prompt = ChatPromptTemplate.from_messages([
			("system", "Use the following context to answer the question. If you don't know, say so."),
			("human", "Context:\n{context}\n\nQuestion:\n{input}")
		])

		# Create chains
		qa_chain = create_stuff_documents_chain(self.llm, prompt)
		retriever = self.vector_store.as_retriever(search_kwargs={"k": 6})
		self.rag_chain = create_retrieval_chain(retriever, qa_chain)

	def ask(self, question: str) -> str:
		"""Ask a question and get an answer"""
		if not self.rag_chain:
			raise ValueError("Chain not setup. Call setup_chain() first.")

		response = self.rag_chain.invoke({"input": question})
		return response.get("answer", "No answer found.")

def main():
	# Initialize RAG
	rag = SimpleRAG("Generative-AI-and-LLMs-for-Dummies.pdf")

	# Process documents
	print("Loading documents...")
	num_chunks = rag.load_and_process_documents()
	print(f"Created {num_chunks} document chunks")

	# Setup chain
	print("Setting up RAG chain...")
	rag.setup_chain()

	# Ask question
	question = "What are Large Language Models?"
	print(f"\nQuestion: {question}")
	answer = rag.ask(question)
	print(f"\nAnswer: {answer}")

if __name__ == "__main__":
	main()