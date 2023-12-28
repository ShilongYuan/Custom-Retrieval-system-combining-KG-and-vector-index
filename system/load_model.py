from answer_generator import Answer_Generator
from rewriter import Request_Rewriter
from passage_extractor import Passage_Extractor
from fact_checker import Fact_Checker
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, AutoConfig
from retriever import *

def get_model(model):
    if model=='chatgpt':
        model = 'chatgpt'
        tokenizer =''
    elif model == 'llama2':
        tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf', trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-chat-hf', trust_remote_code=True).half().cuda()
        model = model.eval()

    elif model=='chatglm':
        tokenizer = AutoTokenizer.from_pretrained('THUDM/chatglm3-6b', trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained('THUDM/chatglm3-6b', trust_remote_code=True).half().cuda()
        model = model.eval()
    if tokenizer.pad_token_id is not None:
            model.config.pad_token_id = tokenizer.pad_token_id
    else:
        model.config.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    passage_extractor = Passage_Extractor(model, tokenizer)

    fact_checker = Fact_Checker(model, tokenizer)

    
    searcher = index_retriever()
    
    request_rewriter = Request_Rewriter(model, tokenizer)
    
    answer_generator = Answer_Generator(model, tokenizer)

    return request_rewriter, searcher, passage_extractor, answer_generator, fact_checker
