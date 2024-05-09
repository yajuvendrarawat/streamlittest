import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import CTransformers
from langchain.llms import Replicate
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import torch
import os
from dotenv import load_dotenv
import tempfile
# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    pipeline,
)

def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! Ask me anything about 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey! 👋"]

def conversation_chat(query, chain, history):
    result = chain({"question": query, "chat_history": history})
    history.append((query, result["answer"]))
    return result["answer"]

def display_chat_history(chain):
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Ask about your Documents", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            with st.spinner('Generating response...'):
                output = conversation_chat(user_input, chain, st.session_state['history'])

            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")

def create_conversational_chain(vector_store):
    load_dotenv()
    
    # Create llm
    #llm = CTransformers(model="llama-2-7b-chat.ggmlv3.q4_0.bin",
                        #streaming=True, 
                        #callbacks=[StreamingStdOutCallbackHandler()],
                        #model_type="llama", config={'max_new_tokens': 500, 'temperature': 0.01})
    #llm = Replicate(
    #    streaming = True,
    #    model = "yajuvendra/Llama-2-7b-chat-finetune", 
    #    callbacks=[StreamingStdOutCallbackHandler()],
    #    input = {"temperature": 0.01, "max_length" :500,"top_p":1})
    #memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    #chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type='stuff',
    #                                             retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
    #                                            memory=memory)
    return chain

def main():
    load_dotenv()
    torch.cuda.is_available()
    model_name = "yajuvendra/Llama-2-7b-chat-finetune"
    embedding_model_name= "sentence-transformers/all-MiniLM-L6-v2"
    # Load the entire model on the GPU 0
    device_map = {"": 0}
    
    # Initialize session state
    initialize_session_state()
    st.title("Multi-Docs ChatBot using llama2 :books:")
    # Initialize Streamlit
    st.sidebar.title("Document Processing")
    uploaded_files = st.sidebar.file_uploader("Upload files", accept_multiple_files=True)    

    if uploaded_files:
        text = []
        for file in uploaded_files:
            file_extension = os.path.splitext(file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(file.read())
                temp_file_path = temp_file.name

            loader = None
            if file_extension == ".pdf":
                loader = PyPDFLoader(temp_file_path)
            elif file_extension == ".docx" or file_extension == ".doc":
                loader = Docx2txtLoader(temp_file_path)
            elif file_extension == ".txt":
                loader = TextLoader(temp_file_path)

            if loader:
                text.extend(loader.load())
                os.remove(temp_file_path)

        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=100, length_function=len)
        text_chunks = text_splitter.split_documents(text)

        # Create embeddings
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name, 
                                           model_kwargs={'device': 'cuda:0'})

        # Create vector store
        vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)

        retriever = vector_store.as_retriever()
        template = """Use the following pieces of context to answer the question at the end. If you don't know the answer,\
        just say that you don't know, don't try to make up an answer.
    
        {context}
    
        {history}
        Question: {question}
        Helpful Answer:"""
    
        prompt = PromptTemplate(input_variables=["history", "context", "question"], template=template)
        memory = ConversationBufferMemory(input_key="question", memory_key="history")

        # Load LLaMA tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=device_map
        )

        pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=1000)
        
        #Create the chain object
        #chain = create_conversational_chain(vector_store)
        llm = HuggingFacePipeline(pipeline=pipe)

        qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt, "memory": memory},
        )
        # Interactive questions and answers
        while True:
            query = input("\nEnter a query: ")
            if query == "exit":
                break
            # Get the answer from the chain
            res = qa(query)
            answer, docs = res["result"], res["source_documents"]
    
            # Print the result
            print("\n\n> Question:")
            print(query)
            print("\n> Answer:")
            print(answer)
    
            if show_sources:  # this is a flag that you can set to disable showing answers.
                # # Print the relevant sources used for the answer
                print("----------------------------------SOURCE DOCUMENTS---------------------------")
                for document in docs:
                    print("\n> " + document.metadata["source"] + ":")
                    print(document.page_content)
                print("----------------------------------SOURCE DOCUMENTS---------------------------")
    

        #display_chat_history(chain)

if __name__ == "__main__":
    main()
