import os
import requests
from tqdm import tqdm
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import re
from langchain_core.documents import Document

# --- CONFIGURAÇÃO ---
DATA_PATH = "data/raw"
DB_FAISS_PATH = "vectorstores/db_faiss"


SOURCES = [
    {
        "name": "constituicao_federal.pdf",
        "url": "https://www2.senado.leg.br/bdsf/bitstream/handle/id/685819/CF88_EC135_2025_separata.pdf",
        "pretty_name": "Constituição Federal de 1988"
    },
    {
        "name": "codigo_defesa_consumidor.pdf",
        "url": "https://www2.senado.leg.br/bdsf/bitstream/handle/id/533814/cdc_e_normas_correlatas_2ed.pdf",
        "pretty_name": "Código de Defesa do Consumidor"
    }
]
# --------------------

def process_pdf_with_article_metadata(path: str, pretty_name: str, text_splitter: RecursiveCharacterTextSplitter):
    """
    Carrega um PDF e o processa para extrair artigos como metadados para cada chunk.
    """
    print(f"Processando com metadados: {pretty_name}...")
    
    # Carrega o PDF inteiro, mas mantém as páginas separadas
    loader = PyPDFLoader(path)
    pages = loader.load_and_split()
    
    all_chunks = []
    current_article_text = "Não especificado"
    
    # Regex para encontrar "Art. Xº" ou "Art. Xo" ou "Art. X."
    article_pattern = re.compile(r"(Art\.\s*\d+[ºo]?)")

    for page in pages:
        content = page.page_content
        page_number = page.metadata.get('page', 0) + 1 # PyPDFLoader começa a página 0
        
        # Encontra todos os inícios de artigo na página
        matches = list(article_pattern.finditer(content))
        
        if not matches:
            # Se não houver novos artigos na página, todos os chunks pertencem ao último artigo visto
            chunks = text_splitter.split_text(content)
            for chunk_content in chunks:
                new_doc = Document(
                    page_content=chunk_content,
                    metadata={
                        "source": path,
                        "pretty_name": pretty_name,
                        "article": current_article_text,
                        "page": page_number
                    }
                )
                all_chunks.append(new_doc)
            continue

        # Se houver artigos na página, processa o conteúdo entre eles
        start_index = 0
        for i, match in enumerate(matches):
            # O texto antes do artigo atual ainda pertence ao artigo anterior
            before_article_content = content[start_index:match.start()]
            if before_article_content.strip():
                chunks = text_splitter.split_text(before_article_content)
                for chunk_content in chunks:
                     all_chunks.append(Document(
                        page_content=chunk_content,
                        metadata={
                            "source": path, "pretty_name": pretty_name, 
                            "article": current_article_text, "page": page_number
                        }
                    ))
            
            # Atualiza o artigo atual
            current_article_text = match.group(1).strip()
            start_index = match.start()

            # Se este for o último artigo encontrado na página, o resto da página pertence a ele
            if i == len(matches) - 1:
                 remaining_content = content[start_index:]
                 chunks = text_splitter.split_text(remaining_content)
                 for chunk_content in chunks:
                     all_chunks.append(Document(
                        page_content=chunk_content,
                        metadata={
                            "source": path, "pretty_name": pretty_name, 
                            "article": current_article_text, "page": page_number
                        }
                    ))

    print(f"Documento '{pretty_name}' dividido em {len(all_chunks)} chunks com metadados de artigo.")
    return all_chunks

def download_files():
    """
    Verifica se os arquivos de dados existem e, caso contrário, faz o download.
    """
    print("Verificando a existência dos arquivos de dados...")
    os.makedirs(DATA_PATH, exist_ok=True)

    for source in SOURCES:
        file_path = os.path.join(DATA_PATH, source["name"])
        if not os.path.exists(file_path):
            print(f"Arquivo '{source['name']}' não encontrado. Baixando de {source['url']}...")
            try:
                response = requests.get(source["url"], stream=True)
                response.raise_for_status()  # Lança um erro para respostas ruins (4xx ou 5xx)

                total_size = int(response.headers.get('content-length', 0))
                block_size = 1024  # 1 KB

                with open(file_path, 'wb') as f, tqdm(
                    total=total_size, unit='iB', unit_scale=True, desc=source["name"]
                ) as pbar:
                    for data in response.iter_content(block_size):
                        pbar.update(len(data))
                        f.write(data)
                
                if total_size != 0 and pbar.n != total_size:
                    print(f"ERRO: Download de '{source['name']}' incompleto.")
                    os.remove(file_path) # Remove o arquivo parcial
                else:
                    print(f"Download de '{source['name']}' concluído com sucesso.")

            except requests.exceptions.RequestException as e:
                print(f"ERRO ao baixar '{source['name']}': {e}")
                if os.path.exists(file_path):
                    os.remove(file_path) # Limpa arquivos corrompidos/parciais
        else:
            print(f"Arquivo '{source['name']}' já existe. Pulando o download.")


def create_vector_db():
    """
    Cria o banco de dados de vetores a partir dos PDFs na pasta de dados,
    usando a lógica de processamento de metadados de artigo.
    """
    print("\nIniciando a criação do banco de dados de vetores...")

    # Instancia o text_splitter que vamos usar para todos os documentos
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    
    final_documents_with_metadata = []

    # Itera sobre nossas fontes definidas em SOURCES
    for source in SOURCES:
        file_path = os.path.join(DATA_PATH, source["name"])
        if os.path.exists(file_path):
            # Usa nossa nova função de processamento inteligente
            processed_docs = process_pdf_with_article_metadata(
                path=file_path,
                pretty_name=source["pretty_name"],
                text_splitter=text_splitter
            )
            final_documents_with_metadata.extend(processed_docs)
        else:
            print(f"AVISO: Arquivo {source['name']} não encontrado. Pulando a indexação.")

    if not final_documents_with_metadata:
        print("Nenhum documento foi processado. Verifique a pasta de dados e os arquivos.")
        return
        
    print(f"\nTotal de {len(final_documents_with_metadata)} chunks de texto criados com metadados.")

    embeddings_model = HuggingFaceEmbeddings(
        model_name='thenlper/gte-small',
        model_kwargs={'device': 'cpu'}
    )

    print("Criando o índice FAISS e indexando os documentos... Isso pode levar alguns minutos.")
    db = FAISS.from_documents(final_documents_with_metadata, embeddings_model)

    os.makedirs(DB_FAISS_PATH, exist_ok=True)
    db.save_local(DB_FAISS_PATH)
    print(f"Banco de dados de vetores salvo em: {DB_FAISS_PATH}")

if __name__ == '__main__':
    download_files()
    create_vector_db()