import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

PDF_PATH = os.getenv("PDF_PATH", "document.pdf")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_EMBEDDING_MODEL = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/embedding-001")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/rag")
PG_VECTOR_COLLECTION_NAME = os.getenv("PG_VECTOR_COLLECTION_NAME", "pdf_chunks")

def get_embeddings():
    if GOOGLE_API_KEY:
        logger.info(f"Usando Google Generative AI Embeddings - {GOOGLE_EMBEDDING_MODEL}")
        return GoogleGenerativeAIEmbeddings(
            model=GOOGLE_EMBEDDING_MODEL,
            google_api_key=GOOGLE_API_KEY
        )
    elif OPENAI_API_KEY:
        logger.info(f"Usando OpenAI Embeddings - {OPENAI_EMBEDDING_MODEL}")
        return OpenAIEmbeddings(
            model=OPENAI_EMBEDDING_MODEL,
            openai_api_key=OPENAI_API_KEY
        )
    else:
        raise ValueError("É necessário configurar GOOGLE_API_KEY ou OPENAI_API_KEY nas variáveis de ambiente")

def load_and_split_pdf(pdf_path: str):
    logger.info(f"Carregando PDF: {pdf_path}")
    
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Arquivo PDF não encontrado: {pdf_path}")
    
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    logger.info(f"PDF carregado com {len(documents)} páginas")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    
    logger.info(f"Documento dividido em {len(chunks)} chunks")
    
    return chunks

def create_vector_store(chunks, embeddings):
    logger.info("Criando vector store no PostgreSQL")
    
    try:
        vector_store = PGVector.from_documents(
            documents=chunks,
            embedding=embeddings,
            connection_string=DATABASE_URL,
            collection_name=PG_VECTOR_COLLECTION_NAME,
            pre_delete_collection=True
        )
        
        logger.info("Vector store criado com sucesso")
        return vector_store
        
    except Exception as e:
        logger.error(f"Erro ao criar vector store: {e}")
        raise

def ingest_pdf():
    try:
        logger.info("Iniciando processo de ingestão do PDF")
        
        chunks = load_and_split_pdf(PDF_PATH)
        
        embeddings = get_embeddings()
        
        vector_store = create_vector_store(chunks, embeddings)
        
        logger.info("Processo de ingestão concluído com sucesso!")
        logger.info(f"Total de chunks armazenados: {len(chunks)}")
        
        return vector_store
        
    except Exception as e:
        logger.error(f"Erro durante a ingestão: {e}")
        raise

if __name__ == "__main__":
    ingest_pdf()