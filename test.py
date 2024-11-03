from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import streamlit as st
from wxai_langchain import Credentials
from wxai_langchain.llm import LangChainInterface
import pandas as pd

# Configuración de credenciales para IBM Watson
creds = Credentials(
    api_key='8P1oPXOxD652hdndMpdcWlY1XS652dIt5MWMaKdB8Vf2',
    project_id='28c90629-1042-4103-81c9-40a114861c4b',
    api_endpoint='https://us-south.ml.cloud.ibm.com'
)

# Configuración del modelo de lenguaje
llm = LangChainInterface(
    credentials=creds,
    model='mistralai/mistral-large',
    params={
        'decoding_method': 'sample',
        'max_new_tokens': 300,
        'temperature': 0.3
    },
)

# Función para cargar y procesar el archivo Excel
@st.cache_resource
def load_excel(file):
    df = pd.read_excel(file)

    # Convertir cada fila en un documento para el índice
    documents = []
    for _, row in df.iterrows():
        row_text = ""
        for col_name, cell_value in row.items():
            row_text += f"{col_name}: {cell_value}\n"
        document = Document(page_content=row_text)
        documents.append(document)

    # Crear el índice vectorial
    index = VectorstoreIndexCreator(
        embedding=HuggingFaceEmbeddings(model_name='all-MiniLM-L12-v2'),
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    ).from_documents(documents)

    return index

# Interfaz de usuario de Streamlit
st.title('Talk with Data')

# Carga de archivo de Excel
uploaded_file = st.file_uploader("Sube tu archivo Excel", type="xlsx")

if uploaded_file:
    index = load_excel(uploaded_file)
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=index.vectorstore.as_retriever(),
        input_key='question'
    )

    # Configuración de mensajes de chat en la sesión
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    # Entrada de pregunta en el chat
    prompt = st.chat_input('Escribe tu pregunta aquí')

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        
        # Generación de respuesta usando la cadena de consulta
        response = chain.run(prompt)
        
        st.chat_message('INC Bot').markdown(response)
        st.session_state.messages.append({'role': 'assistant', 'content': response})
else:
    st.info("Por favor, sube un archivo Excel para empezar.")
