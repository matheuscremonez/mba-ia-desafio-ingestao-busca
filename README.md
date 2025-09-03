# Sistema de Ingestão e Busca Inteligente de PDF

Sistema para processar documentos PDF e permitir busca conversacional baseada em IA.

## **Como Executar a Solução**

### **Pré-requisitos**

1. **Python 3.8+** instalado
2. **Docker e Docker Compose** instalados  
3. **API Key** da OpenAI ou Google Gemini

### **Configuração**

1. **Clone e instale dependências:**
```bash
pip install -r requirements.txt
```

2. **Configure variáveis de ambiente** (crie arquivo `.env`):
```bash
# Arquivo PDF
PDF_PATH=document.pdf

# API Keys (configure pelo menos uma)
GOOGLE_API_KEY=your-google-api-key-here
# OU
OPENAI_API_KEY=sk-your-openai-api-key-here

# Banco de dados (padrão funciona com docker-compose)
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/rag

# O restante das variáveis necessárias, utilizar como está no example já preenchido.
```

3. **Inicie o PostgreSQL com pgVector:**
```bash
docker-compose up -d
```

### **Execução**

#### **1. Processar PDF (Ingestão)**
```bash
python src/ingest.py
```

**O que acontece:**
- Divide o PDF em chunks de 1000 caracteres (overlap 150)
- Converte chunks em embeddings (text-embedding-3-small ou models/embedding-001)
- Salva no PostgreSQL com pgVector

#### **2. Iniciar Chat Inteligente**
```bash
python src/chat.py
```

**Interface do chat:**
- Busca semântica nos chunks do PDF
- Responde usando LLM (gpt-5-nano ou gemini-2.5-flash-lite)
- Mostra fontes consultadas
- Interface conversacional amigável

## **Tecnologias Utilizadas**

### **Modelos de IA**

| **Provedor** | **Embeddings** | **LLM** |
|--------------|----------------|---------|
| **Google Gemini** | models/embedding-001 | gemini-2.5-flash-lite |
| **OpenAI** | text-embedding-3-small | gpt-5-nano |

### **Stack Técnica**

- **Python** + **LangChain**: Framework de IA
- **PostgreSQL** + **pgVector**: Banco vetorial
- **PyPDFLoader**: Processamento de PDF
- **RecursiveCharacterTextSplitter**: Divisão inteligente de texto

## **Arquitetura do Sistema**

```
PDF → Ingestão → PostgreSQL →   Busca →   Resposta
         |           |            |          |
      Chunks      Embeddings  Similarity    LLM
    (1000chars)   (Vetores)    Search     Response
```

## **Comandos do Chat**

- `help` - Mostra comandos disponíveis
- `clear` - Limpa a tela  
- `quit` - Encerra o chat
- **Qualquer pergunta** - Busca no PDF e responde

## **Exemplo de Uso**

```bash
$ python src/chat.py

============================================================
CHAT COM PDF - Sistema de Busca Inteligente
============================================================
Faça perguntas baseadas no PDF carregado!
Digite 'quit' para encerrar
Digite 'help' para ver comandos disponíveis
------------------------------------------------------------
Sistema configurado corretamente!
Encontrados documentos no banco de dados
Usando Google Gemini (models/embedding-001 + gemini-2.5-flash-lite)

Sistema iniciado! Pode começar a fazer perguntas.

Você: Qual é o tema principal do documento?

Buscando informações...

=====================================================
RESPOSTA:
=====================================================
O documento aborda principalmente...

=====================================================
FONTES CONSULTADAS:
=====================================================

Fonte 1:
   O tema central discutido é...
   Página: 1
```
