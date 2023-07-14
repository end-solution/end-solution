#coding=utf-8

from langchain import PromptTemplate, LLMChain
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM

model_id = "/home/ubuntu/openai_knowledgebase/t5_large"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=300
)

local_llm = HuggingFacePipeline(pipeline=pipe)

from langchain.document_loaders import UnstructuredFileLoader
#from langchain.loaders import UnstructuredFileLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain import OpenAI


loader = UnstructuredFileLoader("./end.txt")
document = loader.load()
print(f'documents:{len(document)}')

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 300,
    chunk_overlap = 0
)

split_documents = text_splitter.split_documents(document)
print(f'split_documents:{len(split_documents)}')

'''
#alternative
text_splitter = CharacterTextSplitter()

with open("./end.txt") as f:
    state_of_the_union = f.read()
texts = text_splitter.split_text(state_of_the_union)

from langchain.docstore.document import Document

split_documents = [Document(page_content=t) for t in texts]
print(f'split_documents:{len(split_documents)}')
#alternative
'''
prompt_template = """What should you do when enemy come from south? Write a concise summary in less than 100 words according to the following:

{text}

CONCISE SUMMARY IN ENGLISH:"""

from langchain.prompts import PromptTemplate
PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
chain = load_summarize_chain(local_llm, chain_type="stuff", prompt=PROMPT)
print(chain.run(split_documents))
