import os
import requests
from tqdm import tqdm
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- CONFIGURAÇÃO ---
DATA_PATH = "data/raw"
DB_FAISS_PATH = "vectorstores/db_faiss"

# Lista de documentos esperados e suas URLs de origem
# Links diretos para os PDFs do site do Planalto
SOURCES = [
    {
        "name": "constituicao_federal.pdf",
        "url": "https://www2.senado.leg.br/bdsf/bitstream/handle/id/685819/CF88_EC135_2025_separata.pdf"
    },
    {
        "name": "codigo_defesa_consumidor.pdf",
        "url": "https://www2.senado.leg.br/bdsf/bitstream/handle/id/533814/cdc_e_normas_correlatas_2ed.pdf"
    }
    # Adicione mais documentos aqui no futuro
]
# --------------------

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
    Cria o banco de dados de vetores a partir dos PDFs na pasta de dados.
    """
    print("\nIniciando a criação do banco de dados de vetores...")

    loader = DirectoryLoader(DATA_PATH, glob='*.pdf', loader_cls=PyPDFLoader, show_progress=True)
    documents = loader.load()
    if not documents:
        print("Nenhum documento foi carregado. Verifique a pasta de dados e os arquivos.")
        return
    print(f"{len(documents)} documento(s) carregado(s).")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    texts = text_splitter.split_documents(documents)
    print(f"{len(texts)} chunks de texto criados.")

    embeddings_model = HuggingFaceEmbeddings(
        model_name='thenlper/gte-small',
        model_kwargs={'device': 'cpu'}
    )

    print("Criando o índice FAISS e indexando os documentos... Isso pode levar alguns minutos.")
    db = FAISS.from_documents(texts, embeddings_model)

    os.makedirs(DB_FAISS_PATH, exist_ok=True)
    db.save_local(DB_FAISS_PATH)
    print(f"Banco de dados de vetores salvo em: {DB_FAISS_PATH}")

if __name__ == '__main__':
    download_files()
    create_vector_db()