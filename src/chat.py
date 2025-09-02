import os
import sys
from search import search_and_answer
import logging

logging.getLogger().setLevel(logging.ERROR)

def print_header():
    print("=" * 60)
    print("CHAT COM PDF - Sistema de Busca Inteligente")
    print("=" * 60)
    print("Faça perguntas baseadas no PDF carregado!")
    print("Digite 'quit' para encerrar")
    print("Digite 'help' para ver comandos disponíveis")
    print("-" * 60)

def print_help():
    print("\n" + "=" * 40)
    print("COMANDOS DISPONÍVEIS:")
    print("=" * 40)
    print("help      - Mostra esta ajuda")
    print("clear     - Limpa a tela")
    print("quit      - Encerra o chat")
    print("-" * 40)
    print("Para fazer perguntas, apenas digite sua pergunta!")
    print("=" * 40 + "\n")

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def format_response(response):
    if isinstance(response, dict):
        print("\n" + "=" * 53)
        print("RESPOSTA:")
        print("=" * 53)
        print(f"{response['answer']}")
        
        if response.get('sources'):
            print("\n" + "=" * 53)
            print("FONTES CONSULTADAS:")
            print("=" * 53)
            for i, source in enumerate(response['sources'], 1):
                print(f"\nFonte {i}:")
                print(f"   {source['content']}")
                if source.get('metadata'):
                    metadata = source['metadata']
                    if 'page' in metadata:
                        print(f"   Página: {metadata['page']}")
        print("=" * 53)
    else:
        print(f"\n{response}")

def validate_system():
    try:
        google_key = os.getenv("GOOGLE_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")
        
        if not google_key and not openai_key:
            print("ERRO: Nenhuma API key configurada!")
            print("Configure GOOGLE_API_KEY ou OPENAI_API_KEY no arquivo .env")
            return False
        
        from search import get_vector_store
        vector_store = get_vector_store()
        
        test_docs = vector_store.similarity_search("test", k=1)
        if not test_docs:
            print("ERRO: Nenhum documento encontrado no banco!")
            print("Execute primeiro: python src/ingest.py")
            return False
        
        print("Sistema configurado corretamente!")
        print(f"Encontrados documentos no banco de dados")
        
        if google_key:
            google_embedding = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/embedding-001")
            google_llm = os.getenv("GOOGLE_LLM_MODEL", "gemini-2.5-flash-lite")
            print(f"Usando Google Gemini ({google_embedding} + {google_llm})")
        else:
            openai_embedding = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
            openai_llm = os.getenv("OPENAI_LLM_MODEL", "gpt-5-nano")
            print(f"Usando OpenAI ({openai_embedding} + {openai_llm})")
        
        return True
        
    except Exception as e:
        print(f"ERRO na validação: {e}")
        print("Verifique se o PostgreSQL está rodando e o PDF foi processado")
        return False

def main():
    clear_screen()
    print_header()
    
    if not validate_system():
        print("\nSistema não está pronto. Corrija os erros acima e tente novamente.")
        return
    
    print("\nSistema iniciado! Pode começar a fazer perguntas.\n")
    
    while True:
        try:
            user_input = input("Você: ").strip()
            
            if user_input.lower() == 'quit':
                print("\nAté logo! Obrigado por usar o sistema.")
                break
            
            elif user_input.lower() == 'help':
                print_help()
                continue
            
            elif user_input.lower() == 'clear':
                clear_screen()
                print_header()
                continue
            
            elif not user_input:
                print("Por favor, digite uma pergunta ou comando.")
                continue
            
            print("\nBuscando informações...")
            response = search_and_answer(user_input)
            format_response(response)
            print()
            
        except KeyboardInterrupt:
            print("\n\nInterrompido pelo usuário. Até logo!")
            break
        except Exception as e:
            print(f"\nErro inesperado: {e}")
            print("Tente novamente ou digite 'sair' para encerrar.")

if __name__ == "__main__":
    main()