from langchain.document_loaders import WebBaseLoader
from langchain.embeddings import HuggingFaceEmbeddings


loader = WebBaseLoader("https://www.nju.edu.cn")

data = loader.load()
# embeddings = HuggingFaceEmbeddings()
print(data)
# data = data[0].dict()['page_content']
# doc_result = embeddings.embed_documents([data])
# print(doc_result)

