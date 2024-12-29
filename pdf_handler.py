from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from llm_chains import load_vectordb, create_embeddings
from InstructorEmbedding import INSTRUCTOR
import pypdfium2

def get_pdf_texts(pdfs_bytes):
    return [extract_text_from_pdf(pdf_bytes) for pdf_bytes in pdfs_bytes]

def extract_text_from_pdf(pdf_bytes):
    pdf_file = pypdfium2.PdfDocument(pdf_bytes)
    return "\n".join(pdf_file.get_page(page_number).get_textpage().get_text_range() for page_number in range(len(pdf_file)))

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=50, separators=["\n", "\n\n"])
    return splitter.split_text(text)

def get_document_chunks(texts):
    documents = []
    for text in texts:
        if isinstance(text, list):  # If text is a list, join it into a string
            text = " ".join(text)
        elif not isinstance(text, str):
            raise ValueError(f"Unexpected input type: {type(text)}. Expected string or list.")

        # Split the text into chunks if needed, e.g., by paragraph or fixed length
        chunks = text.split("\n\n")  # Example: split by paragraphs
        for chunk in chunks:
            chunk = chunk.strip()  # Remove extra spaces
            if chunk:  # Only add non-empty chunks
                documents.append(Document(page_content=chunk))

    return documents

def add_documents_to_db(pdfs_bytes):
    texts = get_pdf_texts(pdfs_bytes)
    documents = get_document_chunks(texts)
    vector_db = load_vectordb(create_embeddings())
    vector_db.add_documents(documents)