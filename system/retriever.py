import pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings,HuggingFaceEmbeddings,SelfHostedEmbeddings,HuggingFaceBgeEmbeddings
from langchain.document_loaders import WebBaseLoader
from transformers import BertForMaskedLM,PretrainedConfig, AutoTokenizer,pipeline
import openai
import os
import json
import runhouse as rh
gpu = rh.cluster(name="rh-a10x", instance_type="A100:1", use_spot=False)

pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),  # find at app.pinecone.io
    environment=os.getenv("PINECONE_ENV"),  # next to api key in console
)

class index_retriever:
    def __init__(self,index_name,config) -> None:
        self.config =  config
        self.all_docs = self.get_doc_from_folder()
        self.index_name = index_name
        if index_name not in pinecone.list_indexes():
            print('There are no {} index in your pinecone',index_name)
        else:
            self.doc_search = pinecone.Index(index_name)
        if self.config.tem_type == 'openai' and self.language=='en':
            openai.api_key = os.environ.get('OPEN_AI_API_KEY')
            MODEL = "text-embedding-ada-002"
            self.embedding = OpenAIEmbeddings(model = MODEL)
        elif self.config.tem_type =='Sentence_transformer' and self.language=='en':
            self.embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        elif self.config.tem_type == 'train':
            save_path = config.output_dir
            # config = PretrainedConfig.from_pretrained(save_path)
            # tokenizer = AutoTokenizer.from_pretrained(save_path, config=config)
            # model = BertForMaskedLM.from_pretrained(save_path)
            self.embedding = SelfHostedEmbeddings(
                                model_load_fn=get_pipeline(save_path),
                                hardware= gpu,
                                model_reqs=["./", "torch", "transformers"],
                                inference_fn=inference_fn,
                            ) 
    def get_doc_from_folder(self):
        """
        get all docs & properties from a folder containing json file
        """
        files = os.listdir(self.config.url_list)
        files.sort()
        all_data = []
        for file in files:
            print("Loading: ",file)
            file_path = os.path.join(self.config.url_list, file)
            with open(file_path, "r", encoding = "utf-8") as f:
                doc = json.load(f)
            all_data.append(doc)
        return all_data
    def search(self,query):
        embed_query = self.embedding(query)
        response = self.doc_search.query(
            top_k=3,
            vector=embed_query,
            include_values=True,
        )['matches']
        raw_reference_list = []
        for i in range(len(response)):
            doc_id = response[i]['id']
            raw_reference = self.all_docs[doc_id]
            raw_reference_list.append(raw_reference)
        return raw_reference_list
    
def get_pipeline(path):
    config = PretrainedConfig(path)
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = BertForMaskedLM.from_pretrained(path,config=config)
    return pipeline("feature-extraction", model=model, tokenizer=tokenizer)
def inference_fn(pipeline, prompt):
    # Return last hidden state of the model
    if isinstance(prompt, list):
        return [emb[0][-1] for emb in pipeline(prompt)]
    return pipeline(prompt)[0][-1]
        