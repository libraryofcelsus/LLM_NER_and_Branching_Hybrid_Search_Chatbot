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
              top_p=float(top_p),
              max_tokens=int(max_tokens),
              temperature=float(temperature),
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
              max_tokens=int(max_tokens),
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


def chunk_upload(collection_name, extracted_relations):        
    try:
        with open('settings.json', 'r', encoding='utf-8') as f:
            settings = json.load(f)
        OpenAI.api_key = settings['Open_Ai_Key']
        print_debug = settings['Print_Debug']
        upload_memories = settings['Upload_Memories']
        username = settings['Username']
        user_id = settings['User_Id']
        bot_name = settings['Bot_Name']
        print_debug = settings['Print_Debug']

        entities = parse_entities(extracted_relations)
        relations = parse_relations(extracted_relations)
        print("\n\nEXTRACTED DATA:")
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
                    'memory_type': 'Conversation',
                    'source': "Chatbot Response",
                    'string': f"{source_low} {relation_type_low} {target_low}",
                    'Source_Entity_Type': source_low,
                    'Source_Entity': source_low,
                    'Target_Entity': target_low,
                    'Relation': relation_type_low,
                    'context': evidence_low,
                }

                if client:
                    client.upsert(collection_name=collection_name, points=[PointStruct(id=unique_id, payload=metadata, vector=vector1)])
                    if print_debug == "True":
                        print(f"Successfully uploaded relation {relation_type_low} between {source_low} and {target_low}")
            except Exception as e:
                print(f"Error processing relation: {e}")
                traceback.print_exc()
    except Exception as e:
        print(f"Error in chunk_upload: {e}")
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
    print_debug = settings['Print_Debug']
    upload_memories = settings['Upload_Memories']
    username = settings['Username']
    user_id = settings['User_Id']
    bot_name = settings['Bot_Name']
    usernameupper = username.upper()
    botnameupper = bot_name.upper()
    extraction = list()
    conversation = list()
    input_expansion = list()
    memories = list()
    conv_length = 3
    greeting_msg = "Hello, I am your personal assistant.  What can I help you with?"
    main_conversation = MainConversation(username, user_id, bot_name, conv_length, greeting_msg)
    collection_name = f"Hybrid_Search_Example"
    try:
        collection_info = client.get_collection(collection_name=collection_name)
        print(collection_info)
    except:
        client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=embed_size, distance=Distance.COSINE),
    )
    while True:

        conversation_history = main_conversation.get_conversation_history()
        con_hist = '\n'.join(conversation_history)
        timestamp = time()
        timestring = timestamp_to_datetime(timestamp)
        input_expansion = []  
        user_input = input(f'\n{username}: ').strip()
        user_input_end = "" 
        input_expansion.append({'role': 'user', 'content': f"PREVIOUS CONVERSATION HISTORY: {con_hist}\n\n\n"})
        input_expansion.append({'role': 'system', 'content': f"You are a task rephraser. Your primary task is to rephrase the user's most recent input succinctly and accurately. Please return the rephrased version of the user’s most recent input. USER'S MOST RECENT INPUT: {user_input} {user_input_end}"})
        input_expansion.append({'role': 'assistant', 'content': "TASK REPHRASER: Sure! Here's the rephrased version of the user's most recent input: "})
        expanded_input = gpt_4_completion(input_expansion)
        if print_debug == "True":
            print(expanded_input)
        
        conversation.append({'role': 'system', 'content': f"You are {bot_name}, a chatbot designed to answer questions for the user, {username}."})
        
        extraction = list()
        extraction = [{
            'role': 'system',
            'content': """Analyze the user, %s's input to identify and label key entities. These entities will search nodes on a larger Knowledge Graph Structure. These entities include but are not limited to categories such as PERSON, CONCEPT, ORGANIZATION, SKILL, EVENT, KNOWLEDGE DOMAIN, LOCATION, DATE, TIME, and more. These entities will be used to search the chatbot, %s's memory. If no Entities are detected, assume the subjects are %s and %s.

Expected output format:
1. Entities:
- Entity 1: {type: [ENTITY TYPE], value: [ENTITY NAME], description: [EXTRACTED INFORMATION ON ENTITY]}
- Entity 2: {type: [ENTITY TYPE], value: [ENTITY NAME], description: [EXTRACTED INFORMATION ON ENTITY]}
...

Message from %s to %s:""" % (username, bot_name, username, bot_name, username, bot_name)
}]
       
                        
        extraction.append({'role': 'user', 'content': f"ORIGINAL MESSAGE FROM {usernameupper} to {botnameupper}: {user_input}\nEXPANDED USER INPUT: {expanded_input}"})
        extracted_relations = gpt_4_extraction_completion(extraction)
        if print_debug == "True":
            print(extracted_relations)
        extraction.clear()

        pattern = r'\{type: (.*?), value: (.*?), description: (.*?)\}'
        matches = re.findall(pattern, extracted_relations)
        entities = [{'type': match[0], 'value': match[1], 'description': match[2]} for match in matches]
        for entity in entities:
            if print_debug == "True":
                print(f"Type: {entity['type']}, Value: {entity['value']}, Description: {entity['description']}")
            
            vector_input = embeddings(expanded_input)
            conversation_list = list()
            conversation_list.append({'role': 'system', 'content': f"You are {bot_name}. A chatbot with long term memory.  Use your memories to answer the user's inquiries."})
            conversation_list.append({'role': 'assistant', 'content': f"CONVERSATION HISTORY: {con_hist}"})
            conversation_list.append({'role': 'assistant', 'content': f"CHATBOT MEMORIES: "})
            try:
                hits = client.search(
                    collection_name=f"Hybrid_Search_Example",
                    query_vector=vector_input,
                    query_filter=Filter(
                        must=[
                            FieldCondition(
                                key="user",
                                match=models.MatchValue(value=f"{user_id}"),
                            ),
                        ]
                    ),
                    limit=15
                )
                unsorted_table = [(hit.payload['timestring'], hit.payload['context']) for hit in hits]
                sorted_table = sorted(unsorted_table, key=lambda x: x[0]) 
                joined_table = "\n".join([f"{context}" for timestring, context in sorted_table])
                if print_debug == "True":
                    print(f"\n\nPRIMARY DB SEARCH: {joined_table}")
                memories.append({'role': 'assistant', 'content': f"{joined_table}"})
                target_entities = [hit.payload['Target_Entity'] for hit in hits]
                target_entity_list = "\n".join([f"{Target_Entity}" for Target_Entity in target_entities])
                if print_debug == "True":
                    print(f"\n\nLIST OF TARGET NODES FROM PRIMARY SEARCH: {target_entities}")
                for Target_Entity in target_entities:
                    if print_debug == "True":
                        print(f"\n\nTARGET ENTITY: {Target_Entity}\n\n")
                    target_lower = Target_Entity.lower()
                    vector_input = embeddings(Target_Entity)
                    try:
                        hits = client.search(
                            collection_name="Hybrid_Search_Example",
                            query_vector=vector_input,
                            query_filter=Filter(
                                must=[
                                    FieldCondition(
                                        key="user",
                                        match=models.MatchValue(value=f"{user_id}"),
                                    ),
                                    FieldCondition(
                                        key="context",
                                        match=models.MatchText(text=target_lower),
                                    ),
                                ]
                            ),
                            limit=5
                        )
                        unsorted_table2 = [(hit.payload['timestring'], hit.payload['context']) for hit in hits]
                        sorted_table2 = sorted(unsorted_table2, key=lambda x: x[0]) 
                        joined_table2 = "\n".join([f"{context}" for timestring, context in sorted_table2])
                        if print_debug == "True":
                            print(f"SECONDARY DB SEARCH: {joined_table2}")
                        memories.append({'role': 'assistant', 'content': f"{joined_table2}"})
                    except Exception as e:
                        traceback.print_exc()
                
            except Exception as e:
                traceback.print_exc()
        
        def remove_duplicate_dicts(input_list):
            seen = set()
            output_list = []
            for item in input_list:
                serialized_item = json.dumps(item, sort_keys=True)
                if serialized_item not in seen:
                    seen.add(serialized_item)
                    output_list.append(item)
            return output_list

        memories = remove_duplicate_dicts(memories)
        memory_list = ' '.join(memory['content'] for memory in memories)
        print(f"\n\n\nRETRIEVED MEMORIES: {memory_list}")
        
        
        conversation_list.append({'role': 'assistant', 'content': f"{memory_list}"})
        conversation_list.append({'role': 'user', 'content': f"\n\nUse the chatbot's memories to answer the user, {username}'s question."})
        conversation_list.append({'role': 'user', 'content': f"USER INITIAL INQUIRY: {user_input}"})
        conversation_list.append({'role': 'assistant', 'content': f"Sure, here is a response to the user's input based on the given memories: "})

        response = gpt_4_completion(conversation_list)
        print(f"\n\n\nFINAL RESPONSE: {response}")
        conversation_list.clear()
        memories.clear()
        
        if upload_memories == "True":
            
            extraction = list()
            extraction = [{
                'role': 'system',
                'content': """Analyze the provided text to identify and categorize all principal entities.  For each identified entity, assign a descriptive label, and extract detailed information that fully encapsulates the entity's context and significance. Additionally, identify and articulate the relationships between these entities, specifying the nature of each relationship, the entities involved, and providing a clear, concise name for each entity that reflects its full contextual background. The aim is to generate entity and relation descriptions that are precise and comprehensive, suitable for integration into a knowledge graph database schema. Ensure that all entity names are succinct yet descriptive enough to facilitate easy linkage and database optimization. Present only the information pertinent to entities and their relations, omitting any unnecessary details.

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
            extraction.append({'role': 'user', 'content': f"TEXT TO EXTRACT RELATIONS FROM: {response}"})
            extracted_relations = gpt_4_extraction_completion(extraction)
            if print_debug == "True":
                print(extracted_relations)
            else:
                pass
            chunk_upload(collection_name, extracted_relations)

            
       
        print("\n")
        
        