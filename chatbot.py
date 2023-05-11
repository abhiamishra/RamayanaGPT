import streamlit as st
import openai
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from PIL import Image
from streamlit_chat import message
import pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings

import os
img = Image.open('image.png')
st.set_page_config(page_title='RamayanaGPT', page_icon=img)
# Authenticate with Pinecone using your API key
pinecone.init(
    api_key=st.secrets["PINECONE_API_KEY"], # find at app.pinecone.io.
    environment=st.secrets["PINECONE_API_ENV"]
)

QUERY_MODEL_NAME = "text-embedding-ada-002"

openai_embedding = OpenAIEmbeddings()


# Name of the index you want to load embeddings from
index_name = "ramayana"

# Get a handle to the Pinecone index
index = pinecone.Index(index_name)

print(pinecone.describe_index(index_name))

vectorstore = Pinecone(index=index, embedding_function=openai_embedding.embed_query, text_key="text")

llm = OpenAI(temperature=0, openai_api_key=st.secrets["OPENAI_API_KEY"])
question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)
doc_chain = load_qa_chain(llm, chain_type="stuff")

chain = ConversationalRetrievalChain(
        retriever=vectorstore.as_retriever(),
        question_generator=question_generator,
        combine_docs_chain=doc_chain,
        return_source_documents=True
)

# from haystack.nodes import PromptNode
# from haystack.nodes.prompt import PromptNode, PromptTemplate

# openai_api_key = st.secrets["OPENAI_API_KEY"]

# prompt_node = PromptNode(model_name_or_path="gpt-3.5-turbo", api_key=openai_api_key)
# messages = [{"role": "system", "content": "You are a helpful assistant"}]

# def build_chat(user_input: str = "", asistant_input: str = ""):
#   if user_input != "":
#     messages.append({"role": "user", "content": user_input})
#   if asistant_input != "":
#     messages.append({"role": "assistant", "content": asistant_input})

# def generate_response(input: str):
#   build_chat(user_input=input)
#   chat_gpt_answer = prompt_node(messages)
#   build_chat(asistant_input=chat_gpt_answer[0])
#   return chat_gpt_answer

chat_history = []

def generate_response(query, chat_history, option="Ram"):
    result = chain({"question": query, "chat_history": chat_history})
    chat_history = [(query, result["answer"])]

    prompt = PromptTemplate(
    input_variables=["result", "character"],
      template= '''Rewrite the following text: {result} as if spoken by {character} from the Ramayana. Add personality from the {character} given their history and a catchphrase at the end.''',
    )

    from langchain.chains import LLMChain
    llm2 = OpenAI(temperature=0.7, openai_api_key=st.secrets["OPENAI_API_KEY"])
    chain_two = LLMChain(llm=llm2, prompt=prompt)
    print(result)
    r2 = chain_two.run({"result": result, "character": option})
    print(r2)
    return [r2, chat_history]

st.title("RamayanaGPT")
st.image(img)

option = st.selectbox(
         'Which character would you like to use?',
        ('Ram', 'Sita', 'Hanuman', 'Ravan'))

st.write('You selected:', option)

# Storing the chat
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

# We will get the user's input by calling the get_text function
def get_text():
    input_text = st.text_input("You: ","", key="input")
    return input_text

user_input = get_text()

if user_input:
    output = generate_response(user_input, chat_history, option)
    chat_history = output[1]
    # output = generate_response(user_input)

    #store the output
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output[0])

if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
