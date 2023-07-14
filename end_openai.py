#coding=utf-8
import openai
import os
os.environ["OPENAI_API_KEY"] = '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'
openai.api_key = "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"

from langchain.document_loaders import UnstructuredFileLoader
#from langchain.loaders import UnstructuredFileLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain import OpenAI

loader = UnstructuredFileLoader("./end.txt")
document = loader.load()
print(f'documents:{len(document)}')

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
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

llm = OpenAI(model_name="text-davinci-003", max_tokens=1500)

chain = load_summarize_chain(llm, chain_type="refine", verbose=True)

prompt_template = """What should you do when enemy come from south?? Write a concise answer in less in 100 words according to the following:

{text}

CONCISE SUMMARY IN ENGLISH:"""
from langchain.prompts import PromptTemplate
PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
chain = load_summarize_chain(llm, chain_type="stuff", prompt=PROMPT)
print(chain.run(split_documents))
