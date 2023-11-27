import openai
import os

import tempfile
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import base64

from configs import config

os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY
openai.api_key = config.OPENAI_API_KEY

def detect_emo(querry, model='gpt-3.5-turbo'):
    system_prompt = '''You are a emotion detection robot. You are provided a text from a human
                    and your task is to detect which emotion from the folowing list is echoing in the text.
                    The list contains 15 emotions. You can only choose from these 15 and it will be shown on your face as a robot.
                    So you are not allowed to respond with anything other than what is in the list.
                    Here is the list:

                    Resting
                    Cry
                    Grinning
                    Sob
                    Sad
                    Cool
                    Heart Eyes
                    Kisses
                    Sleepy
                    Wink
                    Tongue
                    Lough
                    Angry
                    Confused
                    Talking '''
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': f'{querry}'}
        ],
        temperature=0.0
    )

    return response['choices'][0]['message']['content']


def create_knowledge_base(pdf_path):

    def process_text(text):
        # Split the text into chunks using langchain
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        # Convert the chunks of text into embeddings to form a knowledge base
        embeddings = OpenAIEmbeddings()
        knowledgeBase = FAISS.from_texts(chunks, embeddings)

        return knowledgeBase
    
    pdf_reader = PdfReader(pdf_path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    knowledgeBase = process_text(text)
    return knowledgeBase


def encode_tempfile_to_base64(tempfile):
    with open(tempfile.name, 'rb') as f:
        file_contents = f.read()
        encoded_string = base64.b64encode(file_contents)
    return encoded_string

def mimic3_tts(text):
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as fp:
        os.system(f'mimic3 --cuda --voice {config.SPEAKER} "{text}" > {fp.name}')
        return encode_tempfile_to_base64(fp)

def transcribe(audio):
    audio_file = open(audio, "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    return transcript['text']

def get_response(knowledgeBase, memory, query, input_type='audio', output_type='audio', need_emo=True):
    os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY
    # if input_type == 'audio':
    #     query = transcribe(query)

    prompt = PromptTemplate(
        input_variables=["chat_history", "human_input", "context"], template=config.template
    )

    docs = knowledgeBase.similarity_search(query)
    llm = ChatOpenAI(model_name="gpt-4")
    chain = load_qa_chain(llm, chain_type='stuff', memory=memory, prompt=prompt)

    with get_openai_callback() as cost:
        response = chain({"input_documents": docs, "human_input": query}, return_only_outputs=True)
        response = response['output_text']
        print(cost)

    result = {"response" : response}

    if need_emo:
        emotion = detect_emo(response)
        result['emotion'] = emotion

    # if output_type == 'audio':
    #     audio_response = mimic3_tts(response)
    #     result['audio_response'] = audio_response

    return result


if __name__ == "__main__":

    query = 'what is the title?'
    knowledgeBase = create_knowledge_base(config.sample_pdf_path)
    memory = ConversationBufferMemory(memory_key="chat_history", input_key="human_input")

    get_response(knowledgeBase, memory, query)