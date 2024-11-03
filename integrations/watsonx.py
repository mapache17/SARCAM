from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
load_dotenv()

watsonx_api_key = os.environ["IBM_CLOUD_API_KEY"]
ibm_cloud_url = 'https://us-south.ml.cloud.ibm.com'
project_id = os.environ["PROJECT_ID"]

creds = {
        "url": ibm_cloud_url,
        "apikey": watsonx_api_key 
    }

model = SentenceTransformer('all-mpnet-base-v2')

params = {
    GenParams.DECODING_METHOD: 'greedy',
    GenParams.MIN_NEW_TOKENS: 500,
    GenParams.MAX_NEW_TOKENS: 1500,
    GenParams.RANDOM_SEED: 42,
    GenParams.TEMPERATURE: 0.5,
    GenParams.REPETITION_PENALTY:1,
    GenParams.STOP_SEQUENCES: ["'''\n", "''' "]
}

model = Model(
    model_id='mistralai/mistral-large',
    params=params,
    credentials=creds,
    project_id=project_id)

def send_to_watsonxai(prompt,
                    model_name="mistralai/mistral-large",
                    decoding_method="greedy",
                    max_new_tokens=1500,
                    min_new_tokens=50,
                    repetition_penalty=1,
                    stop_sequences=[]
                    ):
    model_params = {
        GenParams.DECODING_METHOD: decoding_method,
        GenParams.MIN_NEW_TOKENS: min_new_tokens,
        GenParams.MAX_NEW_TOKENS: max_new_tokens,
        GenParams.STOP_SEQUENCES:stop_sequences,
        GenParams.REPETITION_PENALTY: repetition_penalty,
        GenParams.RANDOM_SEED: 42,
        GenParams.TEMPERATURE: 0.5,
        GenParams.STOP_SEQUENCES: ['\'\'\'\n', '\'\'\' ', '\}\'\'\'']
    }
    model = Model(
        model_id=model_name,
        params=model_params,
        credentials=creds,
        project_id=project_id)
    return model.generate_text(prompt)

def define_num_tokens(grouped_text):
    len_text = len(grouped_text)
    base_tokens = 800
    factor = 0.4  
    max_tokens = 2200
    if len_text <= 200:
        tokens_desc = 100
    else:
        tokens_desc = base_tokens + int(factor * len_text)
        tokens_desc = min(tokens_desc, max_tokens)
    return tokens_desc