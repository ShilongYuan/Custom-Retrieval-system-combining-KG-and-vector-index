import os
from llama_index.graph_stores import NebulaGraphStore
from llama_index.storage.storage_context import StorageContext
from llama_index import KnowledgeGraphIndex,
from llama_index import SimpleWebPageReader,Document
from llama_index.vector_stores import PineconeVectorStore
space_name = "graphindex"
import argparse
import runhouse as rh
gpu = rh.cluster(name="rh-a10x", instance_type="A100:1", use_spot=False)
import getpass
os.environ["PINECONE_API_KEY"] = getpass.getpass("Pinecone API Key:")
os.environ["PINECONE_ENV"] = getpass.getpass("Pinecone Environment:")
os.environ["NEBULA_USER"] = getpass.getpass("Nebula user:")
os.environ["NEBULA_PASSWORD"] = getpass.getpass("Passward:")
os.environ[
    "NEBULA_ADDRESS"
] = getpass.getpass("Address like: 127.0.0.1:9669:")
import os
import json
import openai
import pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings,HuggingFaceEmbeddings,SelfHostedEmbeddings,HuggingFaceBgeEmbeddings
from langchain.document_loaders import WebBaseLoader
import pickle
import re
import tqdm
import torch
import subprocess
from transformers import BertForMaskedLM,PretrainedConfig, AutoTokenizer,pipeline
import torch.nn as nn
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
def load_json_data(data_path):
    files = os.listdir(data_path)
    all_data = []
    for file in files:
        print("Loading: ",file)
        file_path = os.path.join(data_path, file)
        with open(file_path, "r", encoding = "utf-8") as f:
            doc = json.load(f)
        all_data.append(doc)
        #all_data += doc
    return all_data
def save_config(args):
    path = args.output_dir+'/config.json'
    with open(path, 'w') as config_file:
        json.dump(args.__dict__, config_file, indent=2)

class Index_Builder:
    def __init__(
            self,
            task_name,
            data_dir,
            cuda_id,
            batch_size,
            language,
            output_dir,
            tem_type,
            # use_vector_db,
            load_by_webloader,
            url_list = None
    ):  
        self.task_name = task_name
        self.url_list = url_list
        self.load_by_web_loader = load_by_webloader
        # self.use_vector_db = use_vector_db
        self.data_dir = data_dir
        self.cuda_id = cuda_id
        self.batch_size = batch_size
        self.language = language
        self.output_dir = output_dir
        self.tem_type = tem_type
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    def train_tem(self):
        pickle_path = self.output_dir+'train.pkl'
        with open(pickle_path,"wb") as f:
            for doc in self.all_data:
         
                doc_content = doc['title'] + doc['contents']
                content = re.sub(r'[\x00-\x1f\x7f-\x9f]', '',doc_content)
                pickle.dump("{}\t{}\n".format(doc['id'],content))
        tem_save_path = self.output_dir+'/trained_tem'
        if self.language == "zh":
            init_model = "bert-base-chinese"
        else:
            init_model = "bert-base-uncased"
        training_args = ["--nproc_per_node", "2",
                            "-m", "train_tem",
                            "--corpus_path", pickle_path,
                            "--output_dir", tem_save_path,
                            "--model_name_or_path", init_model,
                            "--max_seq_length", "512",
                            "--gradient_accumulation_steps", "1",
                            "--per_device_train_batch_size", str(self.batch_size),
                            "--warmup_steps", "1000",
                            "--fp16",
                            "--learning_rate", "2e-5",
                            "--max_steps", "20000",
                            "--dataloader_drop_last",
                            "--overwrite_output_dir",
                            "--weight_decay", "0.01",
                            "--save_steps", "5000",
                            "--lr_scheduler_type", "constant_with_warmup",
                            "--save_strategy", "steps",
                            "--optim", "adamw_torch"]
        subprocess.run(["torchrun"] + training_args)
        return tem_save_path
    def construct_graph(self):
        if self.load_by_web_loader:
            urls = []
            with open(self.url_list) as f:
                for line in f:
                    url = line.strip().split(' ')[1]
                    urls.append(url)
            web_data = []
            for i in urls:    
                web = SimpleWebPageReader(html_to_text=True).load_data(i)
                web_data.append(web[0])
        else:
            self.json_web = load_json_data(self.data_dir)
            web_data = [Document(item) for item in self.json_web]
            doc_title = [item['title'] for item in self.json_web]    
        graph_store = NebulaGraphStore(
        space_name=space_name,
        # edge_types=edge_types,
        # rel_prop_names=rel_prop_names,
        # tags=tags,
        )    
        storage_context = StorageContext.from_defaults(graph_store=graph_store)
        kg_index = KnowledgeGraphIndex.from_documents(
            web_data,
            storage_context=storage_context,
            max_triplets_per_chunk=10,
            space_name=space_name,
            # edge_types=edge_types,
            # rel_prop_names=rel_prop_names,
            # tags=tags,
            include_embeddings=True,
        )
        kg_index.storage_context.persist(persist_dir=self.output_dir)
    def construct_vector_index(self):
        index_save_path = self.output_dir+'/tem.index'
        #load data
        if self.load_by_web_loader:
            urls = []
            with open(self.url_list) as f:
                for line in f:
                    url = line.strip().split(' ')[1]
                    urls.append(url)
            web_data = []
            for i in urls:    
                web = WebBaseLoader(i)
                data = web.load()
                web_data.append(data[0].dict())
            
            doc_content = [item['metadata']['title'] + item['page_content'] for item in web_data]
            doc_title = [item['title'] for item in self.web_data]
        else:
            self.json_web = load_json_data(self.data_dir)
            doc_content = [item['title'] + item['contents'] for item in self.json_web]
            doc_title = [item['title'] for item in self.json_web]
        if self.tem_type == 'openai' and self.language=='en':
            openai.api_key = os.environ.get('OPEN_AI_API_KEY')
            MODEL = "text-embedding-ada-002"
            embedding = OpenAIEmbeddings(model = MODEL)
        elif self.tem_type =='Sentence_transformer' and self.language=='en':
            embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        elif self.tem_type == 'train':
            save_path = self.train_tem()
            # config = PretrainedConfig.from_pretrained(save_path)
            # tokenizer = AutoTokenizer.from_pretrained(save_path, config=config)
            # model = BertForMaskedLM.from_pretrained(save_path)
            embedding = SelfHostedEmbeddings(
                                model_load_fn=get_pipeline(save_path),
                                hardware= gpu,
                                model_reqs=["./", "torch", "transformers"],
                                inference_fn=inference_fn,
                            )
        docs = []
        for doc,title in zip(doc_content,doc_title):
            output = embedding.embed_documents([doc])
            docs.append(tuple([title,output]))
            
        dim = len(docs[0][1])
        print("Finish converting embeddings.")
        pinecone.init(
            api_key=os.getenv("PINECONE_API_KEY"),
            environment=os.getenv("PINECONE_ENV")
        )
        index_name = self.task_name
        # First, check if our index already exists. If it doesn't, we create it
        if index_name not in pinecone.list_indexes():
            # we create a new index
            pinecone.create_index(name=index_name, metric="cosine", dimension=dim)
        index = pinecone.Index("quickstart")
        
        index.upsert(vectors=docs,batch_size=100)
        
        print("Finish building vector index.")
        save_config(args)
        
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', type=str, help="The name your index to store in pinecone")
    parser.add_argument('--data_dir',type=str,help='your html/json path')
    parser.add_argument('--cuda_id',default=0)
    parser.add_argument('--batch_size',default=64)
    parser.add_argument('--language',default='zh')
    parser.add_argument('--output_dir',type=str)
    parser.add_argument('--tem_type',type=str,default='train')
    parser.add_argument('--url_list',type=str,help='id and url json file')
    args = parser.parse_args()
    index_builder = Index_Builder(task_name=args.task_name,
                                  data_dir=args.data_dir,
                                  cuda_id=args.cuda_id,
                                  batch_size=args.batch_size, 
                                  url_list=args.url_list                          
                                  )
    index_builder.construct_vector_index()
    index_builder.construct_graph()