import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_postgres import PGVector
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "gpt-5-nano")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_EMBEDDING_MODEL = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/embedding-001")
GOOGLE_LLM_MODEL = os.getenv("GOOGLE_LLM_MODEL", "gemini-2.5-flash-lite")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/rag")
PG_VECTOR_COLLECTION_NAME = os.getenv("PG_VECTOR_COLLECTION_NAME", "pdf_chunks")

PROMPT_TEMPLATE = """
CONTEXTO:
{context}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{question}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""

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

def get_llm():
    if GOOGLE_API_KEY:
        logger.info(f"Usando Google Generative AI LLM - {GOOGLE_LLM_MODEL}")
        return ChatGoogleGenerativeAI(
            model=GOOGLE_LLM_MODEL,
            google_api_key=GOOGLE_API_KEY,
            temperature=0
        )
    elif OPENAI_API_KEY:
        logger.info(f"Usando OpenAI LLM - {OPENAI_LLM_MODEL}")
        return ChatOpenAI(
            model=OPENAI_LLM_MODEL,
            openai_api_key=OPENAI_API_KEY,
            temperature=0
        )
    else:
        raise ValueError("É necessário configurar GOOGLE_API_KEY ou OPENAI_API_KEY nas variáveis de ambiente")

def get_vector_store():
    try:
        embeddings = get_embeddings()
        
        vector_store = PGVector(
            connection_string=DATABASE_URL,
            embedding_function=embeddings,
            collection_name=PG_VECTOR_COLLECTION_NAME
        )
        
        logger.info("Conectado ao vector store com sucesso")
        return vector_store
        
    except Exception as e:
        logger.error(f"Erro ao conectar com vector store: {e}")
        raise

def search_prompt(question=None):
    try:
        vector_store = get_vector_store()
        
        llm = get_llm()
        
        prompt = PromptTemplate(
            template=PROMPT_TEMPLATE,
            input_variables=["context", "question"]
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "k": 10,
                    "score_threshold": 0.7
                }
            ),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
        
        logger.info("Chain de busca criada com sucesso")
        return qa_chain
        
    except Exception as e:
        logger.error(f"Erro ao criar chain de busca: {e}")
        return None

def search_and_answer(question: str):
    try:
        logger.info(f"Processando pergunta: {question}")
        
        qa_chain = search_prompt()
        
        if not qa_chain:
            return "Erro: Não foi possível inicializar o sistema de busca."
        
        result = qa_chain.invoke({"query": question})
        
        answer = result["result"]
        sources = result.get("source_documents", [])
        
        logger.info(f"Resposta gerada com {len(sources)} fontes")
        
        return {
            "answer": answer,
            "sources": [
                {
                    "content": doc.page_content[:200] + "...",
                    "metadata": doc.metadata
                }
                for doc in sources
            ]
        }
        
    except Exception as e:
        logger.error(f"Erro durante busca e resposta: {e}")
        return f"Erro: {e}"