import sys
import os
from openai import OpenAI
import json
import time
import threading
import concurrent.futures
from time import time, sleep
from datetime import datetime
from uuid import uuid4
import requests
import shutil
from qdrant_client import QdrantClient
from qdrant_client.models import (Distance, VectorParams, PointStruct, Filter, FieldCondition, 
                                 Range, MatchValue)
from qdrant_client.http import models
import numpy as np
import re
import traceback
from PyPDF2 import PdfReader
from ebooklib import epub
from bs4 import BeautifulSoup
import pytesseract
import subprocess
from PIL import Image
from sentence_transformers import SentenceTransformer


model = SentenceTransformer('all-mpnet-base-v2')
embed_size = 768

def embeddings(query):
    vector = model.encode([query])[0].tolist()
    return vector


def check_local_server_running():
    try:
        response = requests.get("http://localhost:6333/dashboard/")
        return response.status_code == 200
    except requests.ConnectionError:
        return False

# Check if local server is running
if check_local_server_running():
    client = QdrantClient(url="http://localhost:6333")
else:
    try:
        with open('settings.json', 'r', encoding='utf-8') as f:
            settings = json.load(f)
        url = settings['Qdrant_URL']
        api_key = settings['Qdrant_API_Key']
        client = QdrantClient(url=url, api_key=api_key)
        client.recreate_collection(
            collection_name="Ping",
            vectors_config=VectorParams(size=1, distance=Distance.COSINE),
        )
    except:
        print("\n\nQdrant is not started.  Please enter API Keys or run Qdrant Locally.")
        sys.exit()
        
with open('settings.json', 'r', encoding='utf-8') as f:
    settings = json.load(f)
api_key = settings['Open_Ai_Key']

client2 = OpenAI(api_key=f'{api_key}')

def gpt_4_completion(query):
    max_counter = 7
    counter = 0
    with open('settings.json', 'r', encoding='utf-8') as f:
        settings = json.load(f)
        
    temperature = settings['Temperature']
    top_p = settings['Top_P']
    max_tokens = settings['Max_Tokens']
    openai_model = settings['Open_Ai_Model']
    
    while True:
        try:
            completion = client2.chat.completions.create(
              model=openai_model,
              top_p=0.8,
              max_tokens=800,
              temperature=0.85,
              messages=query
            )
            response = (completion.choices[0].message.content)
            return response
        except Exception as e:
            counter +=1
            if counter >= max_counter:
                print(f"Exiting with error: {e}")
                exit()
            print(f"Retrying with error: {e} in 20 seconds...")
            sleep(20)
            
            
def gpt_4_extraction_completion(query):
    max_counter = 7
    counter = 0
    with open('settings.json', 'r', encoding='utf-8') as f:
        settings = json.load(f)
        
    max_tokens = settings['Max_Tokens']
    openai_model = settings['Open_Ai_Model']
    
    while True:
        try:
            completion = client2.chat.completions.create(
              model=openai_model,
              top_p=0.8,
              max_tokens=1000,
              temperature=0.7,
              messages=query
            )
            response = (completion.choices[0].message.content)
            return response
        except Exception as e:
            counter +=1
            if counter >= max_counter:
                print(f"Exiting with error: {e}")
                exit()
            print(f"Retrying with error: {e} in 20 seconds...")
            sleep(20)           
            
            
def timestamp_to_datetime(timestamp):
    return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
            
            
def process_files():
    if not os.path.exists('Upload/TXT'):
        os.makedirs('Upload/TXT')
    if not os.path.exists('Upload/TXT/Finished'):
        os.makedirs('Upload/TXT/Finished')
    if not os.path.exists('Upload/PDF'):
        os.makedirs('Upload/PDF')
    if not os.path.exists('Upload/PDF/Finished'):
        os.makedirs('Upload/PDF/Finished')
    if not os.path.exists('Upload/EPUB'):
        os.makedirs('Upload/EPUB')
    if not os.path.exists('Upload/VIDEOS'):
        os.makedirs('Upload/VIDEOS')
    if not os.path.exists('Upload/VIDEOS/Finished'):
        os.makedirs('Upload/VIDEOS/Finished')
    if not os.path.exists('Upload/EPUB/Finished'):
        os.makedirs('Upload/EPUB/Finished')  
    while True:
        try:
            timestamp = time()
            timestring = timestamp_to_datetime(timestamp)
            process_files_in_directory('./Upload/SCANS', './Upload/SCANS/Finished')
            process_files_in_directory('./Upload/TXT', './Upload/TXT/Finished')
            process_files_in_directory('./Upload/PDF', './Upload/PDF/Finished')
            process_files_in_directory('./Upload/EPUB', './Upload/EPUB/Finished')
            process_files_in_directory('./Upload/VIDEOS', './Upload/VIDEOS/Finished')  
            pass
        except:
            traceback.print_exc()
            
            
def process_files_in_directory(directory_path, finished_directory_path, chunk_size=400, overlap=40):
    try:
        files = os.listdir(directory_path)
        files = [f for f in files if os.path.isfile(os.path.join(directory_path, f))]
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for file in files:
                executor.submit(process_and_move_file, directory_path, finished_directory_path, file, chunk_size, overlap)
    except Exception as e:
        print(e)
        traceback.print_exc() 
        
        
def process_and_move_file(directory_path, finished_directory_path, file, chunk_size, overlap):
    try:
        file_path = os.path.join(directory_path, file)
        chunk_text_from_file(file_path, chunk_size, overlap)
        finished_file_path = os.path.join(finished_directory_path, file)
        shutil.move(file_path, finished_file_path)
    except Exception as e:
        print(e)
        traceback.print_exc()
        
def chunk_text(text, chunk_size, overlap):
    chunks = []
    start = 0
    end = chunk_size
    while end <= len(text):
        chunks.append(text[start:end])
        start += chunk_size - overlap
        end += chunk_size - overlap
    if end > len(text):
        chunks.append(text[start:])
    return chunks
        
        
def chunk_text_from_file(file_path, chunk_size=400, overlap=40):
    with open('settings.json', 'r', encoding='utf-8') as f:
        settings = json.load(f)
    OpenAI.api_key = settings['Open_Ai_Key']
    username = settings['Username']
    user_id = settings['User_Id']
    bot_name = settings['Bot_Name']
    try:
        print("Reading given file, please wait...")
        pytesseract.pytesseract.tesseract_cmd = '.\\Tesseract-ocr\\tesseract.exe'
        textemp = None
        file_extension = os.path.splitext(file_path)[1].lower()
        
        texttemp = None 
        
        if file_extension == '.txt':
            with open(file_path, 'r') as file:
                texttemp = file.read().replace('\n', ' ').replace('\r', '')
                
        elif file_extension == '.pdf':
            with open(file_path, 'rb') as file:
                pdf = PdfReader(file)
                texttemp = " ".join(page.extract_text() for page in pdf.pages)
                
        elif file_extension == '.epub':
            book = epub.read_epub(file_path)
            texts = []
            for item in book.get_items_of_type(9):  # type 9 is XHTML
                soup = BeautifulSoup(item.content, 'html.parser')
                texts.append(soup.get_text())
            texttemp = ' '.join(texts)
            
        elif file_extension in ['.png', '.jpg', '.jpeg']:
            image = open_image_file(file_path)
            if image is not None:
                texttemp = pytesseract.image_to_string(image).replace('\n', ' ').replace('\r', '')
                
        elif file_extension in ['.mp4', '.mkv', '.flv', '.avi']:
            audio_file = "audio_extracted.wav"  
            subprocess.run(["ffmpeg", "-i", file_path, "-vn", "-acodec", "pcm_s16le", "-ac", "1", "-ar", "44100", "-f", "wav", audio_file])
            
            model_stt = whisper.load_model("tiny")
            transcribe_result = model_stt.transcribe(audio_file)
            if isinstance(transcribe_result, dict) and 'text' in transcribe_result:
                texttemp = transcribe_result['text']
            else:
                print("Unexpected transcribe result")
                texttemp = ""  
            os.remove(audio_file)  
            
        else:
            print(f"Unsupported file type: {file_extension}")
            return []

        texttemp = '\n'.join(line for line in texttemp.splitlines() if line.strip())
        chunks = chunk_text(texttemp, chunk_size, overlap)
        filelist = list()

        collection_name = f"Hybrid_Search_Example"
        try:
            collection_info = client.get_collection(collection_name=collection_name)
            print(collection_info)
        except:
            client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=embed_size, distance=Distance.COSINE),
        )
        
        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = []
                for chunk in chunks:
                    future = executor.submit(
                        wrapped_chunk_from_file,
                        chunk, collection_name, bot_name, username, embeddings, client, file_path
                    )
                    futures.append(future)

                results = []
                for future in concurrent.futures.as_completed(futures):
                    results.append(future.result())

                filelist = []

        except Exception as e:
            print(f"An error occurred while executing threads: {e}")
            traceback.print_exc()

        table = filelist
        return table
    except Exception as e:
        print(e)
        traceback.print_exc()
        table = "Error"
        return table  
        
        
        
def wrapped_chunk_from_file(chunk, collection_name, bot_name, username, embeddings, client, file_path):
    try:

        result = summarized_chunk_from_file(chunk, collection_name, bot_name, username, embeddings, client, file_path)

        return result
    except Exception as e:
        print(e)
        traceback.print_exc()
        
        
def summarized_chunk_from_file(chunk, collection_name, bot_name, username, embeddings, client, file_path):
    try:
        filesum = list()
        filelist = list()
        filesum.append({'role': 'system', 'content': "MAIN SYSTEM PROMPT: You are an ai text summarizer.  Your job is to take the given text from a scraped file, then return the text in a summarized article form.  Do not generalize, rephrase, or add information in your summary, keep the same semantic meaning."})
        filesum.append({'role': 'user', 'content': f"SCRAPED ARTICLE: {chunk}"})
        filesum.append({'role': 'system', 'content': "INSTRUCTIONS: Summarize the article without losing any factual knowledge and maintaining full context and information. Only print the truncated article, do not include any additional text or comments."})
        filesum.append({'role': 'assistant', 'content': f"SUMMARIZER BOT: Sure! Here is the summarized article based on the scraped text: "})

        text = chunk
        text = gpt_4_completion(filesum)
        if len(text) < 20:
            text = "No File available"
        filecheck = list()
        filecheck.append({'role': 'system', 'content': f"MAIN SYSTEM PROMPT: You are an agent for an automated text scraping tool. Your task is to decide if the previous Ai Agent scraped the text successfully. The scraped text should contain some form of article, if it does, print 'YES'. If the article was scraped successfully, print: 'YES'.  If the text scrape failed or is a response from the first agent, print: 'NO'.\n\n"})
        filecheck.append({'role': 'user', 'content': f"ORIGINAL TEXT FROM SCRAPE: {chunk}\n\n"})
        filecheck.append({'role': 'user', 'content': f"PROCESSED FILE TEXT: {text}\n\n"})
        filecheck.append({'role': 'system', 'content': f"SYSTEM: You are responding for a Yes or No input field. You are only capible of printing Yes or No. Use the format: [AI AGENT: <'Yes'/'No'>]"})
        filecheck.append({'role': 'assistant', 'content': f"ASSISTANT: "})
        # Filecheck disabled

        fileyescheck = 'yes'
        if 'no file' in text.lower():
            print('---------')
            print('Chunk Failed')
            pass
        if 'no article' in text.lower():
            print('---------')
            print('Chunk Failed')
            pass
        if 'you are a text' in text.lower():
            print('---------')
            print('Chunk Failed')
            pass
        if 'no summary' in text.lower():
            print('---------')
            print('Chunk Failed')
            pass  
        if 'i am an ai' in text.lower():
            print('---------')
            print('Chunk Failed')
            pass                
        else:
            if 'cannot provide a summary of' in text.lower():
                text = chunk
            if 'yes' in fileyescheck.lower():
            
            
                extraction = list()
                extraction = [{
                    'role': 'system',
                    'content': """Analyze the provided text to identify and categorize all principal entities including PERSON, ORGANIZATION, LOCATION, DATE, TIME, MONEY, PERCENT, etc. For each identified entity, assign a descriptive label, and extract detailed information that fully encapsulates the entity's context and significance. Additionally, identify and articulate the relationships between these entities, specifying the nature of each relationship, the entities involved, and providing a clear, concise name for each entity that reflects its full contextual background. The aim is to generate entity and relation descriptions that are precise and comprehensive, suitable for integration into a knowledge graph database schema. Ensure that all entity names are succinct yet descriptive enough to facilitate easy linkage and database optimization. Present only the information pertinent to entities and their relations, omitting any unnecessary details.

Expected output format:
1. Entities:
   - Entity 1: {type: [ENTITY TYPE], value: [ENTITY NAME], description: [EXTRACTED INFORMATION ON ENTITY]}
   - Entity 2: {type: [ENTITY TYPE], value: [ENTITY NAME], description: [EXTRACTED INFORMATION ON ENTITY]}
   ...

2. Relations:
   - Relation 1: {type: [RELATION TYPE], source: [ENTITY SOURCE NAME], target: [ENTITY TARGET NAME], evidence: [FULL CONTEXT OF ENTITY RELATION]}
   - Relation 2: {type: [RELATION TYPE], source: [ENTITY SOURCE NAME], target: [ENTITY TARGET NAME], evidence: [FULL CONTEXT OF ENTITY RELATION]}
   ...
"""

                }]
                extraction.append({'role': 'user', 'content': f"TEXT TO EXTRACT RELATIONS FROM: {text}"})
                extracted_relations = gpt_4_extraction_completion(extraction)
                print(extracted_relations)
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = []
                    future = executor.submit(
                        chunk_upload,
                        file_path, collection_name, extracted_relations
                    )
            
            else:
                print('---------')
                print(f'\n\n\nERROR MESSAGE FROM BOT: {fileyescheck}\n\n\n')  
        
        
        table = filelist
        return table
    except Exception as e:
        print(e)
        traceback.print_exc()
        table = "Error"
        return table    
        
        
def normalize_entity_value(value):
    return value.replace('"', '').strip().lower()
    

def parse_entities(text):
    entities = {}
    entity_pattern = re.compile(r'- Entity \d+: \{type: (.+?), value: "?(.+?)"?, description: "?(.+?)"?\}')
    for match in entity_pattern.finditer(text):
        entity = {
            'type': match.group(1).strip(),
            'value': normalize_entity_value(match.group(2)),
            'description': match.group(3).strip()
        }
        entities[entity['value']] = entity
    return entities

def parse_relations(text):
    relations = []
    relation_pattern = re.compile(r'- Relation \d+: {type: (.+?), source: (.+?), target: (.+?), evidence: (.+?)}')
    for match in relation_pattern.finditer(text):
        relation = {
            'type': match.group(1),
            'source': normalize_entity_value(match.group(2)),
            'target': normalize_entity_value(match.group(3)),
            'evidence': match.group(4)
        }
        relations.append(relation)
    return relations
    
        
def chunk_upload(file_path, collection_name, extracted_relations):        
    try:
        payload = list()
        timestamp = time()
        timestring = timestamp_to_datetime(timestamp)

                
                
        entities = parse_entities(extracted_relations)
        relations = parse_relations(extracted_relations)
        print("Entities:", entities)
        print("Relations:", relations)
        for relation in relations:
            try:
                relation_type = relation['type']
                source = relation['source']
                target = relation['target']
                evidence = relation['evidence']
                
                source_low = source.lower()
                target_low = target.lower()
                evidence_low = evidence.lower()
                relation_type_low = relation_type.lower().replace("_", " ")
            except:
                traceback.print_exc()
        
            try:

                vector1 = embeddings(f"{source_low}\n{relation_type_low}\n{target_low}\n{evidence_low}")

                unique_id = str(uuid4())

                metadata = {
                    'bot': bot_name,
                    'user': user_id,
                    'uuid': unique_id,
                    'timestring': timestamp_to_datetime(time()),
                    'memory_type': 'File_Scrape',
                    'source': file_path,
                    'string': f"{source_low} {relation_type_low} {target_low}",
                    'Source_Entity_Type': source_low,
                    'Source_Entity': source_low,
                    'Target_Entity': target_low,
                    'Relation': relation_type_low,
                    'context': evidence_low,
                }
                 
                
            except Exception as e:
                print(f"Error processing relation: {e}")
        

            try:
                client.upsert(collection_name=collection_name, points=[PointStruct(id=unique_id, payload=metadata, vector=vector1)])
                print(f"Successfully uploaded relation {relation_type_low} between {source_low} and {target_low}")
            except Exception as e:
                print(f"Error uploading to database: {e}")

                traceback.print_exc()
    except:
        traceback.print_exc()
            
            
class MainConversation:
    def __init__(self, username, user_id, bot_name, max_entries, greeting):
        botnameupper = bot_name.upper()
        usernameupper = username.upper()
        self.max_entries = int(max_entries)
        self.file_path = f'./history/{user_id}/{bot_name}_main_conversation_history.json'
        self.file_path2 = f'./history/{user_id}/{bot_name}_main_history.json'
        self.main_conversation = [greeting] 
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        if os.path.exists(self.file_path):
            with open(self.file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.running_conversation = data.get('running_conversation', [])
        else:
            self.running_conversation = []
            self.save_to_file()

    def append(self, timestring, username, usernameupper, user_input, bot_name, botnameupper, response_two):
        entry = []
        entry.append(f"{usernameupper}: [{timestring}] - {user_input}")
        entry.append(f"{botnameupper}: {response_two}")
        self.running_conversation.append("\n\n".join(entry))
        while len(self.running_conversation) > self.max_entries:
            self.running_conversation.pop(0)
        self.save_to_file()

    def save_to_file(self):
        history = self.main_conversation + self.running_conversation

        data_to_save = {
            'main_conversation': self.main_conversation,
            'running_conversation': self.running_conversation
        }

        data_to_save2 = {
            'history': [{'visible': entry} for entry in history]
        }
        with open(self.file_path, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, indent=4)
        with open(self.file_path2, 'w', encoding='utf-8') as f:
            json.dump(data_to_save2, f, indent=4)

    def get_conversation_history(self):
        if not os.path.exists(self.file_path) or not os.path.exists(self.file_path2):
            self.save_to_file()
        return self.main_conversation + ["\n\n".join(entry.split("\n\n")) for entry in self.running_conversation]
        
    def get_last_entry(self):
        if self.running_conversation:
            return self.running_conversation[-1]
        else:
            return None
            
    

    
if __name__ == '__main__':
    with open('settings.json', 'r', encoding='utf-8') as f:
        settings = json.load(f)
    OpenAI.api_key = settings['Open_Ai_Key']
    username = settings['Username']
    user_id = settings['User_Id']
    bot_name = settings['Bot_Name']
    usernameupper = username.upper()
    botnameupper = bot_name.upper()
    while True:
        print("Enter a Number:")
        print("1. Upload and Extract Relation from Files")
        print(f"2. Exit Program")
        mode_selection = input('\nENTER NUMBER CHOICE: ').strip()

        if mode_selection == "1":
            process_files()
            print("Processing complete. Returning to main menu...")
        elif mode_selection == "2":
            print("Exiting program.")
            break
        else:
            print("Invalid option, please try again.")
        print("\n")