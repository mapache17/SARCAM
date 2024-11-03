import streamlit as st
import pandas as pd
import json
import re
from fuzzywuzzy import fuzz
import concurrent.futures
import plotly.express as px
from integrations.watsonx import send_to_watsonxai
from dictionary.synonyms import dict_birads, dict_cutaneas
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from wxai_langchain import Credentials
from wxai_langchain.llm import LangChainInterface

# Credenciales para LangChain
creds = Credentials(
    api_key='8P1oPXOxD652hdndMpdcWlY1XS652dIt5MWMaKdB8Vf2',
    project_id='28c90629-1042-4103-81c9-40a114861c4b',
    api_endpoint='https://us-south.ml.cloud.ibm.com'
)

llm = LangChainInterface(
    credentials=creds,
    model='mistralai/mistral-large',
    params={
        'decoding_method': 'sample',
        'max_new_tokens': 300,
        'temperature': 0.5
    }
)

@st.cache_resource
def load_excel(file):
    df = pd.read_excel(file)
    documents = []
    for _, row in df.iterrows():
        row_text = ""
        for col_name, cell_value in row.items():
            row_text += f"{col_name}: {cell_value}\n"
        document = Document(page_content=row_text)
        documents.append(document)

    index = VectorstoreIndexCreator(
        embedding=HuggingFaceEmbeddings(model_name='all-MiniLM-L12-v2'),
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    ).from_documents(documents)
    
    return index




# Función de extracción para JSON estructurado
def extract_info(text, id_documento):
    prompt_final =  '''*Instrucion*:  Eres un asistente del médico radiologo y tu tarea es analizar el excel con los resultados de mamografías y extraer información que no está estructurada para construir un JSON siguiendo un conjunto de reglas específicas. El documento esta en español y puede contener palabras mal escritas.

*Detalles Clave*:

* Tarea: Debes identificar el ID del paciente, el BIRADS, nodulos, morfología de los nodulos, margenes de los nodulos, densidad del nodulo, presencia de microcalcificaciones, calcificaciones tipicamente benignas, calcificaciones morfologia sospechosa, distribucion calcificaciones, presencia de asimetrias, tipo de asimetría, hallazgos asociados, lateralidad del hallazgo, ID del paciente.
* Language: El texto está en español y pueden haber errores ortográficos.
* Format: Retorna la data estructurada en formato JSON, siguiendo una estructura como la siguiente:
   * **Recuerda revisar SIEMPRE el diccionario para identificar correctamente que valor se deberia poner en el JSON
    {{
        \\"Nodulos\\": "0.No" si menciona que no hay presencia de nodulos o no menciona nodulos, "1.Si" si menciona que hay presencia de nodulos, 
        \\"Morfologia_de_los_nodulos\\": "1. Ovalado" si es Ovalado, "2.Redondo" si es Redondo, "3.Irregular" si es Irregular,  "N/A" si no hay presencia de Nodulos o si no se menciona,
        \\"Margenes_nodulos\\": "1.Circunscritos" si son Circunscritos, "2.Microlobulados" si son Microlobulados, "3.Indistintos" si son Indistintos o mal definidos, "4.Obscurecidos" si son Obscurecidos, Opacos u Oscuros, "5.Espiculados" si son Espiculados, "N/A" si no se menciona,
        \\"Densidad_nodulo\\": "1.Densidad Grasa" si es Densidad Grasa, "2.Hipodenso" si es Baja Densidad o Hipodenso, "3.Isodenso" si tiene Igual Densidad o Isodenso, "4.Hiperdenso" si son de Alta Densidad o Hiperdenso, "N/A" si no se menciona,
        \\"Presencia_microcalcificaciones\\": "0.No" si menciona que no hay Microcalcificaciones o si no menciona Microcalcificaciones Benignas ni de Morfologia Sospechosa, "1.Si" si menciona que hay Microcalcificaciones o si menciona Microcalcificaciones Benignas o si son de Morfologia Sospechosa,
        \\"Calcificaciones_tipicamente_benignas\\": "1.Cutaneas" si son Cutaneas, "2.Vasculares" si son Vasculares, "3.Gruesas" si son Gruesas o Popcorn, "4.Leño o Vara" si son Leño o Vara, "5.Puntiformes o Redondas" si son Redondas o Puntiformes, "6.Anulares" si son Anulares, "7.Distroficas" si son Distroficas, "8.Leche de calcio" si son Leche de Calcio, "9.Suturas" si son Suturas, "N/A" si no se menciona, IMPORTANTE: EN ESTE CAMPO PUEDE HABER MÁS DE UNA CALCIFICACION, PRESENTALAS SEPARADAS POR ",",
        \\"Calcificaciones_morfologia_sospechosa\\": "1.Gruesas Heterogeneas" si son Gruesas Heterogeneas, "2.Amorfas" si son Amorfas, "3.Finas Pleomorficas" si son Finas Pleomorficas, "4.Lineas Fianas o lineas Ramificadas" si son Lineas Fianas o Lineales Ramificadas, "N/A" si no aplica,
        \\"Distribucion_de_las_calcificaciones\\": "1.Difusas" si son Difusas, "2.Regionales" si son Regionales, "3.Agrupadas o Cumulo" si son Agrupadas o en Cumulo, "4.Segmentarias" si son Segmentarias, "5.Lineales" si son Lineales, "N/A" si no se menciona,
        \\"Presencia_de_asimetrias\\": "0.No" si menciona que no hay Asimetrias en las mamas o si no menciona asimetrias en las mamas, o sea que son Simetricas, "1.Si" si menciona que sí hay Asimetrias en las mamas, o sea que no son simétricas,
        \\"Tipo_de_asimetria\\": "1.Asimetria" si hay Asimetria, "2.Asimetria Global" si hay Asimetria Global, "3.Asimetria Focal" si hay Asimetria Focal, "4.Asimetria Focal Evolutiva" si hay Asimetria Focal Evolutiva, "N/A" si no se menciona o no hay asimetria, o sea que son simétricas,
        \\"Hallazgos_asociados\\": "1.Retracción de la Piel" si hay Retraccion de la Piel, "2.Retraccion del pezon" si hay Retraccion del Pezon, "3.Engrosamiento de la Piel" si hay Engrosamiento de la Piel, "4.Engrosamiento Trabecular" si hay Engrosamiento Trabecular, "5.Adenopatias axilares" si hay Adenopatias Axilares, "N/A" si no se menciona, IMPORTANTE: EN ESTE CAMPO PUEDE HABER MÁS DE UN HALLAZGO, PRESENTALOS SEPARADOS POR ",",
        \\"Lateralidad_hallazgo\\": "1.Derecho" si es Derecho, "2.Izquierdo" si es Izquierdo, "3.Bilateral" si es Bilateral o si es Predominante, "N/A" si no se menciona lateralidad,
        \\"BIRADS\\": "0.Estudio Incompleto" si el estudio esta incompleto o no hay imagen radiologica, "1.Resultado Negativo" si el resultado es negativo, "2.Hallazgo Benigno" si el hallazgo es Benigno, "3.Hallazgo probablemente Benigno" si el hallazgo es Probablemente Benigno, "4.Hallazgo Sospechoso" si el hallazgo es Sospechoso, "4A.Escasa Presuncion de Malignidad" si hay Escasa presuncion de Malignidad, "4B.Presuncion Moderada de Malignidad" si hay presuncion Moderada de Malignidad, "4C.Gran Presuncion de Malignidad" si hay Gran presuncion de Malignidad, "5.Hallazgo Sugerente de Malignidad" si hay hallazgo Muy sugerente de Malignidad, "6.Diagnostico Maligno Confirmado" si hay diagnostico Maligno Confirmado por biopsia, IMPORTANTE: EL BIRAD PUEDE APARECER EN NÚMEROS O NUMEROS ROMANOS, DE SER ASÍ, ACOMODA EL DATO A NUMEROS DEL ALFABETO LATINO
    }}

* Sinonimos para cada una de:
    * Utiliza los diccionarios proporcionados para identificar los sinónimos que correspondan a cada una de las palabras. Aquí están los diccionarios:
        - Diccionario de sinónimos para "BIRADS": {dict_birads},
        - Diccionario de sinónimos para "Cutaneas": {dict_cutaneas}
'''
    combined_prompt = prompt_final + str(text) + '\n\nOutput: \'\'\'json{'
    extracted_text = '\'\'\'json{' + send_to_watsonxai(combined_prompt, max_new_tokens=5000)
    pattern = r"'''json\s*(.*?)\s*'''"
    match = re.search(pattern, extracted_text, re.DOTALL)
    if match:
        json_content = match.group(1).strip()
        try:
            structured_data = json.loads(json_content)
            structured_data["ID_DOCUMENTO"] = id_documento
            return structured_data
        except json.JSONDecodeError:
            return {}
    return {}

# Procesamiento en lote
def process_batch(batch):
    return [extract_info(row['ESTUDIO'], row['ID_DOCUMENTO']) for row in batch]

logo_path = 'assets/isdhc.png'  # Reemplaza con la ruta de tu logo
col1, col2, col3 = st.columns([1, 8, 1])  # Ajusta los números para distribuir el espacio
with col1:
    st.image(logo_path, width=100) 

# Interfaz de Streamlit con Tabs
st.title("SARCAM")

st.subheader("Procesamiento de Resultados de Mamografías")

# Definición de Tabs
tab1, tab2 = st.tabs(["Dashboard", "Chat con el Documento"])

with tab1:
    # Carga de archivo y procesamiento de resultados de mamografías
    st.write("Sube un archivo de Excel con los resultados para procesar.")
    uploaded_file = st.file_uploader("Sube tu archivo de Excel (.xls o .xlsx)", type=["xls", "xlsx"])

    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file)
            df_filtered = df[~df['ESTUDIO'].apply(lambda x: fuzz.partial_ratio(str(x).lower(), "arpon") > 80)]
            data = df_filtered.to_dict(orient='records')

            # Procesamiento en lotes
            batch_size = 6
            batches = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
            
            # Procesamiento paralelo
            structured_data = []
            with concurrent.futures.ThreadPoolExecutor() as executor:
                results = list(executor.map(process_batch, batches))

            # Aplanar resultados
            for result in results:
                structured_data.extend(result)

            # Convertir structured_data en DataFrame y combinar con datos originales
            df_results = pd.DataFrame(structured_data)
            df_combined = pd.concat([df_filtered, df_results], axis=1)
            
            # Guardar archivo procesado
            output_filename = 'resultados_mamografias.xlsx'
            df_combined.to_excel(output_filename, index=False)

            # Mostrar botón de descarga
            with open(output_filename, "rb") as file:
                st.download_button(label="Descargar resultados procesados", data=file, file_name=output_filename)
            
            # Visualización de gráficos
            st.subheader("Visualización de Datos")
            
            col1, col2 = st.columns(2)

            # Gráfico de distribución de BIRADS
            with col1:
                if 'BIRADS' in df_combined.columns:
                    birads_count = df_combined['BIRADS'].value_counts()
                    fig_birads = px.bar(
                        birads_count,
                        x=birads_count.index,
                        y=birads_count.values,
                        title="Distribución de BIRADS",
                        labels={'x': 'BIRADS', 'y': 'Cantidad'}
                    )
                    fig_birads.update_layout(width=300, height=300)
                    st.plotly_chart(fig_birads, use_container_width=True)
                if 'Presencia_microcalcificaciones' in df_combined.columns:
                    cal_count = df_combined['Presencia_microcalcificaciones'].value_counts()
                    fig_birads = px.pie(
                        cal_count,
                        values=cal_count.values,
                        names=cal_count.index,
                        title="Presencia de microcalcificaciones",
                    )
                    fig_birads.update_layout(width=300, height=300)
                    st.plotly_chart(fig_birads, use_container_width=True)
                if 'Calcificaciones_tipicamente_benignas' in df_combined.columns:
                    cal_count = df_combined['Calcificaciones_tipicamente_benignas'].value_counts()
                    fig_birads = px.bar(
                        cal_count,
                        x=cal_count.index,
                        y=cal_count.values,
                        title="Calcificaciones tipicamente benignas",
                        labels={'x': 'Calcificaciones benignas', 'y': 'Cantidad'}
                    )
                    fig_birads.update_layout(width=300, height=300)
                    st.plotly_chart(fig_birads, use_container_width=True)
            with col2:
                if 'Nodulos' in df_combined.columns:
                    nodulos_count = df_combined['Nodulos'].value_counts()
                    fig_nodulos = px.pie(
                        nodulos_count,
                        values=nodulos_count.values,
                        names=nodulos_count.index,
                        title="Presencia de Nódulos"
                    )
                    fig_nodulos.update_layout(width=300, height=300)  
                    st.plotly_chart(fig_nodulos, use_container_width=True)

                if 'Densidad_nodulo' in df_combined.columns:
                    densidad_count = df_combined['Densidad_nodulo'].value_counts()
                    fig_densidad = px.bar(
                        densidad_count,
                        x=densidad_count.index,
                        y=densidad_count.values,
                        title="Distribución de Densidad de Nódulos",
                        labels={'x': 'Densidad del Nódulo', 'y': 'Cantidad'}
                    )
                    fig_densidad.update_layout(width=300, height=300)  # Ajusta tamaño del gráfico
                    st.plotly_chart(fig_densidad, use_container_width=True)
                if 'Presencia_de_asimetrias' in df_combined.columns:
                    nodulos_count = df_combined['Presencia_de_asimetrias'].value_counts()
                    fig_nodulos = px.pie(
                        nodulos_count,
                        values=nodulos_count.values,
                        names=nodulos_count.index,
                        title="Presencia de asimetrias"
                    )
                    fig_nodulos.update_layout(width=300, height=300)  
                    st.plotly_chart(fig_nodulos, use_container_width=True)


        except Exception as e:
            st.error(f"Ocurrió un error: {e}")

with tab2:
    st.subheader('Chat con el Documento')
    index = load_excel(uploaded_file) if uploaded_file else None

    if index:
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type='stuff',
            retriever=index.vectorstore.as_retriever(),
            input_key='question'
        )

        if 'messages' not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            st.chat_message(message['role']).markdown(message['content'])

        prompt = st.chat_input('Escribe tu pregunta aquí')

        if prompt:
            st.chat_message('user').markdown(prompt)
            st.session_state.messages.append({'role': 'user', 'content': prompt})
            response = chain.run(prompt)
            st.chat_message('assistant').markdown(response)
            st.session_state.messages.append({'role': 'assistant', 'content': response})
    else:
        st.warning("Por favor, sube un archivo para poder interactuar con el chat.")
