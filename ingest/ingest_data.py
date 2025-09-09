import os
import re
from pathlib import Path
import requests
from tqdm import tqdm
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from typing import List, Dict, Any

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = str(PROJECT_ROOT / "data" / "raw")
DB_FAISS_PATH = str(PROJECT_ROOT / "vectorstores" / "db_faiss")

SOURCES = [
    {
        "file_name": "vademecum.pdf",
        "download_url": "https://www2.senado.leg.br/bdsf/bitstream/handle/id/685991/Vade_mecum_EC134_2024.pdf",
        "pretty_name": "Vade Mecum 2025",
        "parser_type": "vade_mecum",
    },
]

def parse_vade_mecum(file_path: Path) -> List[Document]:
    file_path = Path(file_path)
    loader = PyPDFLoader(str(file_path))
    pages = loader.load()
    full_text = "\n".join([page.page_content for page in pages])

    law_title_regex = r'\n([A-ZÇÃÕÁÉÍÓÚ]{5,}(?:\s+(?:DE|DA|DO)\s+[A-ZÇÃÕÁÉÍÓÚ]+)*\s*)\n'
    law_splits = re.split(law_title_regex, full_text)

    if len(law_splits) < 3:
        law_splits = ["Vade Mecum Completo", full_text]

    article_chunks = []
    for i in range(1, len(law_splits), 2):
        law_title = law_splits[i].strip()
        law_text = law_splits[i+1]
        
        article_splits = re.split(r'(?=Art\.\s\d+º(?:-\w+)*)', law_text)
        if article_splits and len(article_splits[0].strip()) < 100:
            article_splits.pop(0)

        for article_text in article_splits:
            match = re.search(r'Art\.\s(\d+º(?:-\w+)*)', article_text)
            article_number = match.group(1) if match else "N/A"
            article_chunks.append(Document(
                page_content=article_text.strip(),
                metadata={
                    "source_file": file_path.name,
                    "pretty_name": law_title,
                    "article": article_number
                }
            ))
    print(f"  -> Extraídos {len(article_chunks)} artigos.")
    return article_chunks

def parse_simple_law(file_path: Path, source_info: Dict[str, Any]) -> List[Document]:
    file_path = Path(file_path)
    loader = PyPDFLoader(str(file_path))
    pages = loader.load()
    full_text = " ".join([page.page_content for page in pages])
    full_text = re.sub(r'\s*\n\s*', ' ', full_text)

    article_splits = re.split(r'(?=Art\.\s\d+º)', full_text)
    if article_splits and len(article_splits[0].strip()) < 50:
        article_splits.pop(0)

    article_chunks = []
    for article_text in article_splits:
        match = re.search(r'Art\.\s(\d+º(?:-\w+)*)', article_text)
        article_number = match.group(1) if match else "N/A"
        article_chunks.append(Document(
            page_content=article_text.strip(),
            metadata={
                "source_file": file_path.name,
                "pretty_name": source_info.get("pretty_name", file_path.stem),
                "article": article_number
            }
        ))
    print(f"  -> Extraídos {len(article_chunks)} artigos.")
    return article_chunks

# --- FUNÇÃO PRINCIPAL DE INGESTÃO ---

def create_vector_db():
    """
    Função principal que orquestra a ingestão de todos os documentos
    definidos na lista SOURCES, usando o parser apropriado para cada um.
    """
    print("\n--- INICIANDO PROCESSO DE INGESTÃO DE DOCUMENTOS ---")
    all_chunks = []

    for source in SOURCES:
        file_name = source["file_name"]
        file_path = os.path.join(DATA_PATH,file_name)
        download_url = source.get("download_url")
        
        if download_url and not os.path.isfile(file_path):
            print(f"Baixando '{file_name}' de '{download_url}'...")
            try:
                response = requests.get(download_url, stream=True)
                response.raise_for_status()
                total_size = int(response.headers.get('content-length', 0))
                block_size = 1024
                
                with open(file_path, 'wb') as f, tqdm(
                    total=total_size, unit='iB', unit_scale=True, desc=file_name
                ) as pbar:
                    for data in response.iter_content(block_size):
                        pbar.update(len(data))
                        f.write(data)
                print(f"Download de '{file_name}' concluído.")
            except requests.exceptions.RequestException as e:
                print(f"ERRO ao baixar '{file_name}': {e}. Pulando.")
                continue
            
        if not os.path.isfile(file_path):
            print(f"AVISO: Ficheiro '{file_name}' não encontrado em '{DATA_PATH}'. Pulando.")
            continue
        
        print(f"\nProcessando ficheiro: {file_name}")
        parser_type = source.get("parser_type", "simple_law")

        if parser_type == "vade_mecum":
            chunks = parse_vade_mecum(file_path)
        elif parser_type == "simple_law":
            chunks = parse_simple_law(file_path, source)
        else:
            print(f"AVISO: Tipo de parser '{parser_type}' desconhecido. Pulando ficheiro.")
            continue
        
        all_chunks.extend(chunks)

    if not all_chunks:
        print("\nERRO: Nenhum chunk foi criado. Verifique os seus ficheiros de origem e configurações.")
        return

    print(f"\n--- Processo de Parsing concluído. Total de {len(all_chunks)} chunks lógicos gerados. ---")

    embeddings_model = HuggingFaceEmbeddings(
        model_name='thenlper/gte-small',
        model_kwargs={'device': 'cpu'}
    )

    print("Criando o índice FAISS... (Isto pode demorar, dependendo do número de chunks)")
    db = FAISS.from_documents(all_chunks, embeddings_model)

    os.makedirs(DB_FAISS_PATH, exist_ok=True)
    db.save_local(str(DB_FAISS_PATH))
    print(f"\n--- SUCESSO! Banco de dados de vetores salvo em: {DB_FAISS_PATH} ---")

if __name__ == '__main__':
    if not os.path.exists(DB_FAISS_PATH):
        os.makedirs(DB_FAISS_PATH)

    create_vector_db()