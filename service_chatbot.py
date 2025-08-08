import numpy as np
import os
import random
import re
import tempfile
import time
import tiktoken
import uuid


from azure.cosmos import CosmosClient
from azure.storage.blob import BlobServiceClient, ContentSettings
from common import logger, utcnow, CosmosDBTable, get_secret
from chatbot import *
from chatbot_config import get_config
from collections import defaultdict, Counter
from datetime import datetime, timezone
from io import StringIO
from tenacity import retry, stop_after_attempt, wait_fixed
from tqdm import tqdm
import json
from typing import List, Tuple

CONFIG = get_config()
services = CONFIG['SERVICES']
DB_NAME = CONFIG['DB_NAME']
# ---- Cosmos Setup ----
client = CosmosClient(CONFIG['DB_URL'], credential=CONFIG['DB_KEY'])

DOCUMENT_METRICS_CONTAINER = CONFIG['DOCUMENT_METRICS_CONTAINER']
CHAT_METRICS_CONTAINER = CONFIG['CHAT_METRICS_CONTAINER']
SERVICE_DEFINATION_CONTAINER = CONFIG['SERVICE_DEFINATION_CONTAINER']
DOCUMENT_QNA_CONTAINER = CONFIG['DOCUMENT_QNA_CONTAINER']              #Human records
DOCUMENT_QNA_AI_CONTAINER = CONFIG['DOCUMENT_QNA_AI_CONTAINER']        #Ai records


def create_dir_if_not_exists(connection_string, blob_container, directory_path):
  try:
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(blob_container)
    blob_list = container_client.list_blobs(name_starts_with=directory_path)
    directory_exists = any(True for _ in blob_list)

    if not directory_exists:
      # Create a zero-length blob to simulate a directory
      blob_client = container_client.get_blob_client(f"{directory_path}.keep")
      blob_client.upload_blob(b"", overwrite=True)

  except Exception as e:
    raise RuntimeError(f"Failed to create directory or it does not exists: {e}")

def create_folders_in_storage():
    for service in services:
                
        storage = service.get('storage', {})
        base_path = service['service_id']
        faq_input_path = base_path + '/FAQ/Input/'
        faq_output_path = base_path + '/FAQ/Output/'
        faq_archive_path = base_path + '/FAQ/Archive/'
        faq_error_path = base_path + '/FAQ/Error/'
        private_path = base_path + "/private/"
        standard_path = base_path + "/Standard/" 
        
        create_dir_if_not_exists(CONFIG['BLOB_URL'], storage.get('blob_container'), faq_input_path)
        create_dir_if_not_exists(CONFIG['BLOB_URL'], storage.get('blob_container'), faq_output_path)
        create_dir_if_not_exists(CONFIG['BLOB_URL'], storage.get('blob_container'), faq_archive_path)
        create_dir_if_not_exists(CONFIG['BLOB_URL'], storage.get('blob_container'), faq_error_path)
        create_dir_if_not_exists(CONFIG['BLOB_URL'], storage.get('blob_container'), private_path)
        create_dir_if_not_exists(CONFIG['BLOB_URL'], storage.get('blob_container'), standard_path)

def fetch_excel_csv_blobs_via_df(connection_string, container_name, faq_input_path, faq_error_path):
    
    try:
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client(container_name)

        blob_list = container_client.list_blobs(name_starts_with=faq_input_path)
        excel_csv_blobs = [blob for blob in blob_list if blob.name.endswith('.csv')]

        logger.info(f"{len(excel_csv_blobs)} CSV files are present inside Input folder for container: {container_name}")

        file_df_list = []
        for blob in excel_csv_blobs:
            blob_client = container_client.get_blob_client(blob.name)
            blob_data = blob_client.download_blob().readall()
            
            if blob.name.endswith('.csv'):
                try:
                    df = pd.read_csv(BytesIO(blob_data), header=None)

                    if df.shape[1] == 1:
                        df['expected_baseline_answer'] = np.nan

                    columns = ['question', 'expected_baseline_answer']
                    df.columns = columns
                    df.reset_index(inplace=True, drop = True)
                    file_df_list.append((blob.name.split("/")[-1], df))
                
                except Exception as e:
                    logger.error(f"{blob.name} file is not in required format, moving the file to Error folder: {e}")
                    timeStamp = datetime.now(timezone.utc).strftime('%Y_%m_%d_%H_%M_%S')
                    file_name = blob.name.split("/")[-1]
                    processed_file_name = file_name.split(".csv")[0] +"_"+ timeStamp + ".csv"                         
                    move_csv_from_source_to_target(container_name, file_name, processed_file_name, source_dir=faq_input_path, target_dir=faq_error_path)
                    logger.info("taking next file...")
                    continue                

        return file_df_list
    except Exception as e:
        raise RuntimeError(f"Error in fetching CSV files from blob Input folder: {e}")

def get_additional_versions(service_id, version):
    # to check additional version is present in a version 
    for service in services:
        if (service['service_id'] == service_id) and (service['version'] == version):

            if 'additional_versions' in service:
                return service['additional_versions']
    return []


def create_container(container_name: str, user_id: int, session_id: str):
   connection_string = CONFIG['BLOB_URL']
   metadata = {
    "session_id": session_id ,
    "user_id": user_id
    }
   try:
    # Create the BlobServiceClient object
      blob_service_client = BlobServiceClient.from_connection_string(connection_string)

      # Create the container
      container_client = blob_service_client.create_container(name = container_name, metadata=metadata)
      logger.info(f"Container '{container_name}' created successfully.")

   except Exception as e:
      logger.error(f"Error: {e}")
      raise e



def add_human_records_document_faq_container(data: list):
  """Adds the records to the specified Cosmos Db table/container"""
  try:
      for record in data:
          data_record = {"service_id": record['service_id'], "id": record['session_id'], 'time_stamp': record['time_stamp'], 
                         "question": record['question'], "chatbot_answer": record['chatbot_answer'], "expected_baseline_answer": record['expected_baseline_answer'], 
                  "version": record['version'], "response_source": "Human", "metrics": record['metrics'], "file_source": record['file_source']}
          
          CosmosDBTable(CONFIG['DB_URL'], CONFIG['DB_KEY'], CONFIG['DB_NAME'], DOCUMENT_QNA_CONTAINER).add(data_record)

  except Exception as e:
      raise RuntimeError(f"Issues while Adding record to the Azure Container")

def upload_csv_to_blob(blob_container, df, directory_path, filename):
    blob_client = None
    try:
        # Create a blob service client
        blob_service_client = BlobServiceClient.from_connection_string(CONFIG['BLOB_URL'])
        container_client = blob_service_client.get_container_client(blob_container)

        # Convert DataFrame to CSV in memory
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        # Create full blob path
        blob_path = f"{directory_path.rstrip('/')}/{filename}"

        # Upload the blob
        container_client.upload_blob(
            name=blob_path,
            data=csv_buffer.getvalue(),
            overwrite=True,
            content_settings=ContentSettings(content_type='text/csv')
        )

        return logger.info(f"Upload successful: {filename}")
    except Exception as e:
        raise RuntimeError(f"Upload failed to Ouput directory: {str(e)}")

def move_csv_from_source_to_target(blob_container, file_name, processed_file_name, source_dir, target_dir):
    try:
        # Create the BlobServiceClient object
        blob_service_client = BlobServiceClient.from_connection_string(CONFIG['BLOB_URL'])
        container_client = blob_service_client.get_container_client(blob_container)

        source_blob_path = source_dir + file_name
        target_blob_path = target_dir + processed_file_name

        source_blob_client = container_client.get_blob_client(source_blob_path)
        blob_data = source_blob_client.download_blob().readall()

        # Upload the content to the target blob
        target_blob_client = container_client.get_blob_client(target_blob_path)
        target_blob_client.upload_blob(blob_data, overwrite=True)

        # Delete the source blob
        source_blob_client.delete_blob()

    except Exception as e:
        raise RuntimeError(f"Error in moving file {file_name} from Input to Archive: {e}")

def add_output_file_dict(output_file_dict, question, expected_baseline_answer, version,chatbot_answer):
    # Generating output file by each record.
    output_file_dict['question'].append(question)
    output_file_dict['expected_baseline_answer'].append(expected_baseline_answer)
    output_file_dict['version'].append(version)
    output_file_dict['chatbot_answer'].append(chatbot_answer['answer'])
    output_file_dict['citation_1_url'].append(chatbot_answer['citations'][0]['url'] if 0 < len(chatbot_answer['citations']) else '')
    output_file_dict['citation_1_document_id'].append(chatbot_answer['citations'][0]['document_id'] if 0 < len(chatbot_answer['citations']) else '')
    output_file_dict['citation_2_url'].append(chatbot_answer['citations'][1]['url'] if 1 < len(chatbot_answer['citations']) else '')
    output_file_dict['citation_2_document_id'].append(chatbot_answer['citations'][1]['document_id'] if 1 < len(chatbot_answer['citations']) else '')
    output_file_dict['citation_3_url'].append(chatbot_answer['citations'][2]['url'] if 2 < len(chatbot_answer['citations']) else '')
    output_file_dict['citation_3_document_id'].append(chatbot_answer['citations'][2]['document_id'] if 2 < len(chatbot_answer['citations']) else '')
    
    return output_file_dict

def faq_processor():
    """
    This subroutine generates processes FAQ in CSV format inside Input folder 
    and generates a processed report inside Output folder.
    Later moves CSV present in Input to Archive folder.
    """

    for service_id in set([service['service_id'] for service in services]): # Runs for each service_id
        
        service = [service for service in services if service['service_id']==service_id][0]
        faq_input_path = service_id + '/FAQ/Input/'
        faq_output_path = service_id + '/FAQ/Output/'
        faq_archive_path = service_id + '/FAQ/Archive/'
        faq_error_path = service_id + '/FAQ/Error/'

        storage = service['storage']
        file_df_list = fetch_excel_csv_blobs_via_df(CONFIG['BLOB_URL'], storage.get('blob_container'), faq_input_path, faq_error_path)
        versions = [service['version'] for service in services if service['service_id'] == service_id and service['version']!='Standard']

        if len(versions) == 0:
           logger.info(f"No versions found for service_id:{service_id}")
           continue
        
        llm = FISLLM()
        for file_name, df in file_df_list:

            logger.info(f"{file_name}: started processing...")
            file_list = [] # list of dictionary to store into CosmosDB filewise
            output_file_dict = {
                'question': [],
                'expected_baseline_answer': [],
                'version': [],
                'chatbot_answer': [],
                'citation_1_url': [],
                'citation_1_document_id': [],
                'citation_2_url': [],
                'citation_2_document_id': [],
                'citation_3_url': [],
                'citation_3_document_id': []
            }

            for version in versions:

                logger.info(f"processing for version: {version}")
                additional_versions = get_additional_versions(service_id, version) # return additional_versions list
                for i, row in df.iterrows():
                    
                    time_stamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
                    session_id = str(uuid.uuid4())
                    question = row['question']
                    expected_baseline_answer = row['expected_baseline_answer']

                    test_length = 20 if len(question)>20 else len(question)
                    logger.info(f"processing question {i}: {question[:test_length]}...")
                    try:
                        chatbot_answer = get_answer(session_id = session_id, service_id = service_id, version = version, query = question, additional_versions = additional_versions)
                    except Exception as e:
                        chatbot_answer = {'answer': "Could not generate answer, please retry in next run", 'citations': []}
                        logger.info(f"Chabot not providing answer. Added default chatbot_answer and skipping this record: {e}")
                        time.sleep(1)


                    file_list.append({
                        'service_id': service_id,
                        'session_id': session_id,
                        'version': version,
                        'time_stamp': time_stamp,
                        'question': question,
                        'expected_baseline_answer': "" if pd.isna(row['expected_baseline_answer']) else expected_baseline_answer,
                        'file_source': file_name,
                        'chatbot_answer': [{'timestamp': time_stamp, 'answer': chatbot_answer}], 
                        'metrics': [] if pd.isna(row['expected_baseline_answer']) else [{'timestamp': time_stamp, 'metric': Metrics(llm.tokenizer)(expected_baseline_answer, chatbot_answer['answer'])}]
                    })

                    output_file_dict = add_output_file_dict(output_file_dict, question, expected_baseline_answer, version, chatbot_answer)
                    
            add_human_records_document_faq_container(file_list)
            logger.info("Records added for file {} and service_id: {}".format(file_name, service_id))

            timeStamp = datetime.now(timezone.utc).strftime('%Y_%m_%d_%H_%M_%S')
            processed_file_name = file_name.split(".csv")[0] +"_"+ timeStamp + ".csv"
            blob_container = storage.get('blob_container')
            
            upload_csv_to_blob(blob_container, pd.DataFrame(output_file_dict), faq_output_path, processed_file_name)
            logger.info("File: {} uploaded to Ouput directory for service_id: {}".format(file_name, service_id))

            move_csv_from_source_to_target(blob_container, file_name, processed_file_name, source_dir=faq_input_path, target_dir=faq_archive_path)
            logger.info("File: {} moved file from Input to Archive for service_id: {}".format(file_name, service_id))

def get_service_provider_details(client_id:str)->dict :
    """
    Fetches service provider details from the IDP using secure credentials.
    """
    try:
        
        service_data=get_secret(client_id)
        return service_data
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Error fetching IDP service provider: {e}", exc_info=True)
        raise RuntimeError("Failed to fetch service provider details from IDP.")

def create_knowledge_base(llm:LLM, service_id:str, version:str) -> KnowledgeBase:
  """
  Creates vector database as a knowledge base.

  :param llm: Underlying LLM with embeddings for text vectorization.
  :param service_id: Service ID, which is used as the first part of index name.
  :param version: Service version, which is used as the second part of index name.
  """
  return CognitiveSearch(url=CONFIG['VDB_URL'], key=CONFIG['VDB_KEY'], index=f'{service_id}_{version}', llm=llm)

def prepare_knowledge_base(
  service_id:str,
  version:str,
  aws_bucket:str,
  aws_region:str,
  azure_container:str,
  folder:str,
  confluence_space:str,
  last_modified:datetime,
  confluence_url:str,
  confluence_token:str,
):
  """
  Fetches files from AWS and/or Azure, extracts text, and updates the knowledge base.
  The files are fetched from AWS/Azure if all aws/azure parameters are set.

  :param service_id: Service ID, which is used as the first part of index name.
  :param version: Service version, which is used as the second part of index name.
  :param aws_bucket: AWS S3 bucket.
  :param aws_region: AWS S3 region.
  :param azure_container: Azure Blob container.
  :param folder: AWS S3 or Azure Blob folder.
  """

  old_urls = set()
  supported_urls = set()
  unsupported_urls = set()
  missing_urls = set()
  def create_skip_function(kb:KnowledgeBase):
    for url in kb.urls:
      if '/$web/' in url:
        # This is a workaround for help manuals that are not hosted on their own website.
        url = f'azure://{azure_container}/' + url.split('/$web/')[1].split('?')[0]
      old_urls.add(url)
      missing_urls.add(url)
    def skip(url):
      if url in old_urls:
        missing_urls.remove(url)
        return True # The file is already processed.
      if not TextLoader.supports(url):
        unsupported_urls.add(url)
        return True
      supported_urls.add(url)
      return False
    return skip

  def create_aws_loaders(skip) -> list[FileLoader]:
    ready = aws_bucket and folder and aws_region
    if not ready:
      return []
    try:
      logger.info(f"🔄 {service_id} ({version}) Fetching files from AWS S3: {aws_bucket}/{folder}, Region: {aws_region}")
      return FileLoaderAWS.from_storage(aws_bucket, folder, aws_region, skip)
    except Exception as e:
      logger.error(f"🚩 {service_id} ({version}) Error fetching files from AWS S3: {e}")
      return []

  def create_azure_loaders(skip) -> list[FileLoader]:
    ready = azure_container and folder
    if not ready:
      return []
    try:
      logger.info(f"🔄 {service_id} ({version}) Fetching files from Azure Blob storage: {azure_container}/{folder}")
      return FileLoaderAzure.from_storage(CONFIG['BLOB_URL'], azure_container, folder, skip, last_modified)
    except Exception as e:
      logger.error(f"🚩 {service_id} ({version}) Error fetching files from Azure Blob storage: {e}")
      return []

  def update_knowledge_base(kb:KnowledgeBase, loader:FileLoader=None, confluence_file:str=False):
    logger.info(f"📡 {service_id} ({version}) Updating knowledge base with: {loader.url}")

    try:
      start_time = time.time()
      if confluence_file:
        loader = TextExtractorConfluence(confluence_url, confluence_token, confluence_space, last_modified)
        file = FileObject(loader.url)
        chunks = loader.extract(kb)
      else:
        file = loader()
        chunks = TextLoader.from_file(file)
      end_time = time.time()
      load_seconds = round(end_time - start_time, 4)
      if not [chunk for chunk in chunks if chunk.text.strip()]:
        logger.info(f"⏭ {service_id} ({version}) Skipping empty file: {loader.url}")
        return 'empty', 0
      
      start_time = time.time()
      chunks = kb.add(*chunks)
      end_time = time.time()
      save_seconds = round(end_time - start_time, 4)
      tokens = kb.llm.count_tokens(chunks)
      text = '\n'.join([c.text for c in chunks])
      
      metrics = CosmosDBTable(CONFIG['DB_URL'], CONFIG['DB_KEY'], CONFIG['DB_NAME'], DOCUMENT_METRICS_CONTAINER)
      metrics_data = {
        'url':loader.url,
        'service_id':service_id,
        'version':version,
        'load_seconds':load_seconds,
        'save_seconds':save_seconds,
        'bytes':None if file.content is None else len(file.content),
        'chunks':len(chunks),
        'tokens':tokens,
        'chars':len(text),
        'words':len(re.sub(r'[^\w\s]', '', text).split()),
        'lines':len(text.splitlines()),
        'timestamp':utcnow().isoformat(),
      }
      for m in metrics:
        if m['url']==loader.url:
          metrics_data['id'] = m['id']
          break
      metrics.add(metrics_data)
      
      logger.info(f"✅ {service_id} ({version}) Successfully updated knowledge base with: {loader.url} [used {tokens} tokens]")
      return 'ingested', tokens
    except Exception as e:
      logger.error(f"🚩 {service_id} ({version}) Failed to process: {loader.url}. {e}", exc_info=True)
      return 'failed', 0

  def log(update_results:list[tuple[str,int]], update_seconds:float) -> bool:
    # Counting the statuses and tokens:
    status2count = {}
    total_token_count = 0
    for status, token_count in update_results:
      status2count[status] = status2count.get(status, 0) + 1
      total_token_count += token_count

    # Preparing counts for logging:
    old_files_count = len(old_urls)
    unsupported_files_count = len(unsupported_urls)
    missing_files_count = len(missing_urls)
    new_files_count = unsupported_files_count + len(supported_urls)
    empty_files_count = status2count.get('empty', 0)
    failed_files_count = status2count.get('failed', 0)
    ingested_files_count = status2count.get('ingested', 0)

    # Preparing additional log info:
    info = []
    info.append(f'{old_files_count} old files')
    info.append(f'{new_files_count} new files')
    if missing_files_count > 0:
      info.append(f'{missing_files_count}/{old_files_count} removed')
    if failed_files_count > 0:
      info.append(f'{failed_files_count}/{new_files_count} failed')
    if unsupported_files_count > 0:
      info.append(f'{unsupported_files_count}/{new_files_count} unsupported')
    if empty_files_count > 0:
      info.append(f'{empty_files_count}/{new_files_count} empty')
    if ingested_files_count > 0:
      info.append(f'{ingested_files_count}/{new_files_count} ingested')
    if total_token_count > 0:
      info.append(f'used {total_token_count} tokens in {update_seconds} seconds')

    # Logging the summary:
    if failed_files_count == 0:
      logger.info(f"✅ {service_id} ({version}) is up to date [{', '.join(info)}]")
      return True # Success
    else:
      logger.warning(f"⚠️ {service_id} ({version}) is not up to date [{', '.join(info)}]")
      return False # Failure

  def test_chatbot(kb:KnowledgeBase) -> bool:
    try:
      bot = Chatbot.assistant(kb.llm, kb, Chat.from_memory(), service_id)
      answer = bot(f'Describe {service_id} in one sentence.')
      usage = f"[used {answer['usage']['input']} input and {answer['usage']['output']} output tokens]"
      citations = str(answer['citations']) if answer['citations'] else ''
      answer = f"{answer['answer']} {usage} {citations}"
      logger.info(f"🤖 {service_id} ({version}) summary: {answer}")
      return True
    except Exception as e:
      logger.error(f"🚩 {service_id} ({version}) Failed to test chatbot: {e}", exc_info=True)
      return False

  def prepare():
    logger.info(f"🚀 {service_id} ({version}) Preparing knowledge base...")
    try:
      # Creating the knowledge base:
      kb = create_knowledge_base(FISLLM(), service_id, version)

      # Creating file loaders:
      skip = create_skip_function(kb)
      file_loaders = create_aws_loaders(skip) + create_azure_loaders(skip)
      if unsupported_urls:
        unsupported_ext2count = {}
        for unsupported_url in unsupported_urls:
          unsupported_ext = FileObject(unsupported_url).ext
          unsupported_ext2count[unsupported_ext] = unsupported_ext2count.get(unsupported_ext, 0) + 1
        unsupported_ext_count = ', '.join([f'{c} x {e}' for e, c in unsupported_ext2count.items()])
        logger.warning(f"⚠️ {service_id} ({version}) Skipping {len(unsupported_urls)} unsupported files: {unsupported_ext_count}")
      if missing_urls:
        logger.info(f"🔄 {service_id} ({version}) Removing {len(missing_urls)} missing files: {', '.join(missing_urls)}")
        for missing_url in missing_urls:
          if not confluence_url or not missing_url.startswith(confluence_url):
            kb.remove(missing_url)
      if not file_loaders and not confluence_space:
        logger.info(f"✅ {service_id} ({version}) is up to date [no new files].")
        return True

      logger.info(f"🔄 {service_id} ({version}) Processing {len(supported_urls)} new files: {', '.join(supported_urls)}")

      update_results = []
      # Updating the knowledge base:
      start_time = time.time()


      for loader in tqdm(file_loaders):
        update_results.append(update_knowledge_base(kb, loader=loader))
      if confluence_space:
        update_results.append(update_knowledge_base(kb, confluence_file=True))
      end_time = time.time()
      update_seconds = round(end_time - start_time, 4)

      # Testing the chatbot:
      return log(update_results, update_seconds) and test_chatbot(kb)
    except Exception as e:
      logger.error(f"🚩 {service_id} ({version}) Failed to prepare knowledge base: {e}", exc_info=True)
      return False

  return prepare()

def faq_driver():
  try:
    logger.info("Batch Started\nChecking FAQ folders in each containers...")
    create_folders_in_storage()
    logger.info("Started processing CSVs located at each service containers...")
    faq_processor()
    return True
  except Exception as e:
    logger.error(f"FAQ processing falied due to error: {e}")    
    return False

def prepare_knowledge_bases():
  services = CONFIG['SERVICES']
  schedule = CosmosDBTable(CONFIG['DB_URL'], CONFIG['DB_KEY'], CONFIG['DB_NAME'], "Document_Logs")
  if not services:
    logger.error("🚩 Failed to prepare knowledge bases. No service configurations!")
    return

  logger.info(f"🚀 Preparing knowledge bases for {len(services)} services. Configuration: {services}")
  failed = []

  # Ensure all existing entries without status are marked as Completed
  for entry in schedule.data():
    for service_id, details in entry.items():
      if isinstance(details, dict) and 'version' in details and 'status' not in details:
        details['status'] = 'Completed'
        schedule.add(entry)

  for service in services:
    last_started = utcnow().strftime('%Y-%m-%d %H:%M:%S%z')
     
    service_id = service.get("service_id")
    if service_id is None:
      logger.error(f"🚩 Missing `service_id` in service config. Skipping service...")
      continue
    version = service.get("version")
    if version is None:
      logger.error(f"🚩 {service_id} Missing `version` in service config. Skipping service...")
      continue
    storage = service.get("storage")
    if storage is None:
      logger.error(f"🚩 {service_id} ({version}) Missing `storage` in service config. Skipping service...")
      continue
    
    schedule_data = {'action':'prepare'}
    for sd in schedule.data():
       if sd.get('action') == 'prepare':
          schedule_data = sd
          break

    service_schedule = schedule_data[service_id] = schedule_data.get(service_id, {})

    s3_bucket = storage.get("s3_bucket", '')
    aws_region = storage.get("aws_region", "us-east-1")
    blob_container = storage.get("blob_container", '')
    input_folder = storage.get("input_folder", '')
    confluence_space = storage.get('space_name', '')
    confluence_url = storage.get('confluence_url', '')
    confluence_token = storage.get('confluence_token', '')
    laststarted = service_schedule.get('laststarted', '')
    last_modified = datetime.strptime(laststarted, '%Y-%m-%d %H:%M:%S%z') if laststarted else None
    try:
      # Mark as In Progress
      service_schedule['status'] = 'In Progress'
      schedule.add(schedule_data)

      if not prepare_knowledge_base(
        service_id=service_id,
        version=version,
        aws_bucket=s3_bucket,
        aws_region=aws_region,
        azure_container=blob_container,
        folder=input_folder,
        confluence_space=confluence_space,
        last_modified=last_modified,
        confluence_url=confluence_url,
        confluence_token=confluence_token,
      ):
        failed.append({'service_id': service_id, "version": version})
      else:
        service_schedule['version'] = version
        service_schedule['laststarted'] = last_started
        schedule.add(schedule_data)
    finally:
      # Mark as Completed regardless of success/failure
      service_schedule['status'] = 'Completed'
      schedule.add(schedule_data)

  if failed:
    logger.warning(f"⚠️ Failed to prepare knowledge bases for {len(failed)} of {len(services)} services: {failed}.")
  else:
    logger.info(f"✅ Successfully prepared knowledge bases for {len(services)} of {len(services)} services.")
  
  return failed

@retry(wait = wait_fixed(2), stop = stop_after_attempt(30))
def get_answer(session_id:str, service_id:str, version:str, query:str, t:float=0.1, additional_versions=[]):
  """Returns the answer to the query from the service assistant."""
  basic_versions = [
     {'service_id': 'FIS', 'version': 'Standard'},
     {'service_id': service_id, 'version': 'Standard'},
     {'service_id': service_id, 'version': version},
  ]

  service_versions = set()
  service_definitions = list(CosmosDBTable(CONFIG['DB_URL'], CONFIG['DB_KEY'], CONFIG['DB_NAME'], SERVICE_DEFINATION_CONTAINER))

  def add_version(v:dict):
    _service = v.get('service_id','')
    _version = v.get('version','')
    _pair = (_service, _version)
    if _service and _version and _pair not in service_versions:
       service_versions.add(_pair)
       find_additional_versions(_service, _version)

  def find_additional_versions(service_id, version):
    for _sd in service_definitions:
       if _sd['service_id']==service_id and _sd['version']==version:
          for _av in _sd.get('additional_versions', {}):
             add_version(_av)

  for v in basic_versions + additional_versions:
    add_version(v)

  llm = FISLLM()
  kb = KnowledgeBases(*[create_knowledge_base(llm, s, v)  for s, v in service_versions])
  chat = Chat.from_cosmosdb(session=session_id, url=CONFIG['DB_URL'], key=CONFIG['DB_KEY'], db=CONFIG['DB_NAME'])
  bot = Chatbot.assistant(llm, kb, chat, service_id)
  start_time = time.time()
  answer = bot(query, t)
  end_time = time.time()    
  seconds = round(end_time - start_time, 4)
  for citation in answer['citations']:
    citation['url'] = FileLoaderAzure.to_public_url(CONFIG['BLOB_URL'], citation['url']) + f"#page={citation['pages']}"
    
  answer['session_id'] = session_id
  answer['answer_id'] = answer_id = str(uuid.uuid4())
  sessions = CosmosDBTable(CONFIG['DB_URL'], CONFIG['DB_KEY'], CONFIG['DB_NAME'], CHAT_METRICS_CONTAINER)
  sessions.add({
    'answer_id':answer_id,
    'session_id':session_id,
    'service_id':service_id,
    'version':version,
    'query':query,
    'answer':answer,
    'seconds':seconds,
    'input_tokens': answer['usage']['input'],
    'output_tokens': answer['usage']['output'],
    'timestamp':utcnow().isoformat(),
  })
  return answer

def add_feedback(service_id:str, session_id:str, answer_id:str, feedback:str, comment:str, rating:str):
  """Adds feedback for the last user query and answer."""
  chat = Chat.from_cosmosdb(session=session_id, url=CONFIG['DB_URL'], key=CONFIG['DB_KEY'], db=CONFIG['DB_NAME'])
  query = ''
  answer = ''
  for message in chat.messages:
    if message.get('role') == 'user':
      query = message.get('content', '')
    if message.get('role') == 'assistant':
      answer = message.get('content', '')
  sessions = CosmosDBTable(CONFIG['DB_URL'], CONFIG['DB_KEY'], CONFIG['DB_NAME'], "ChatBot_Sessions")
  query = f"SELECT * FROM c WHERE c.answer_id = '{answer_id}'"
  items = list(sessions.client.query_items(query=query, enable_cross_partition_query=True))
  updated = False
  for item in items:
     if item.get('answer_id') == answer_id:
        if feedback:
          item['feedback']=feedback
        if comment:
          item['comment']=comment
        if rating:
           item['rating'] = rating
        sessions.client.replace_item(item=item['id'], body=item)
        updated = True

  if not updated:
    sessions.add({
      'service_id':service_id,
      'answer_id':answer_id,
      'session_id':session_id,
      'query':query,
      'answer':answer,
      'feedback':feedback,
      'comment':comment,
      'rating':rating,
      'timestamp':utcnow().isoformat(),
    })

def view_records_by_service_id(service_id: str, version: str = None):
    """Returns all records from the specified Cosmos DB table/container based on service_id, Version(Optional)"""
    container = client.get_database_client(DB_NAME).get_container_client(SERVICE_DEFINATION_CONTAINER)
    if version:
        query = f"SELECT * FROM c WHERE c.service_id = '{service_id}' AND c.version = '{version}'"
    else:
        query = f"SELECT * FROM c WHERE c.service_id = '{service_id}'"
    items = list(container.query_items(query=query, enable_cross_partition_query=True))
    if not items:
        raise Exception(f"No records found for service ID: {service_id}" + (f" and version: {version}" if version else ""))
    return items

def view_all_records_chatbot_service_defination() -> list[dict]:
    """Returns all records from the specified Cosmos DB table/container."""
    records = list(CosmosDBTable(CONFIG['DB_URL'], CONFIG['DB_KEY'], CONFIG['DB_NAME'],SERVICE_DEFINATION_CONTAINER))
    return records

def add_record_chatbot_service_defination(session_id: str, service_id: str, version: str, 
                                           blob_container : str, input_folder: str, confluence_space:str,
                                             confluence_url:str, confluence_token: str, additional_versions:list[dict]=[]):
    """Adds the records to the specified Cosmos DB table/container."""
    try:
        # Check if the record already exists
        existing_records = view_records_by_service_id(service_id, version)
        if existing_records:
            raise Exception(f"Record with service_id '{service_id}' and version '{version}' already exists.")
    except Exception:
        # Proceed to add the record if not found
        storage = {"blob_container": blob_container, "input_folder": input_folder,
                   "confluence_space": confluence_space, "confluence_url": confluence_url, "confluence_token": confluence_token}
        data = {"service_id": service_id, "version": version, "storage": storage, "additional_versions": additional_versions, "id": session_id}

        CosmosDBTable(CONFIG['DB_URL'], CONFIG['DB_KEY'], CONFIG['DB_NAME'], SERVICE_DEFINATION_CONTAINER).add(data)

def update_chatbot_service_defination_records(service_id: str, blob_container: str, input_folder: str,
                                              confluence_space: str, confluence_url: str, confluence_token: str, version: str,
                                              additional_versions:list[dict]=[]):
    """Update `storage` in the record based on the container(table_name) & version, preserving existing values if new ones are empty."""
    
    # New values from UI
    new_storage = {
        "blob_container": blob_container,
        "input_folder": input_folder,
        "confluence_space": confluence_space,
        "confluence_url": confluence_url,
        "confluence_token": confluence_token
    }

    container = client.get_database_client(DB_NAME).get_container_client(SERVICE_DEFINATION_CONTAINER)
    query = f"SELECT * FROM c WHERE c.service_id = '{service_id}' AND c.version = '{version}'"
    items = list(container.query_items(query=query, enable_cross_partition_query=True))

    if not items:
        raise Exception(f"No record found for service ID '{service_id}' with version '{version}'.")

    for item in items:
        existing_storage = item.get('storage', {})
        # Merge: keep existing values if new ones are empty
        updated_storage = {
            key: new_storage[key] if new_storage[key] else existing_storage.get(key, "")
            for key in new_storage
        }
        item['storage'] = updated_storage
        if additional_versions is not None:
          item['additional_versions'] = additional_versions
        container.replace_item(item=item['id'], body=item)

def delete_chatbot_service_defination_by_service_id(service_id:str, version:str):
    """Delete the record based on the service_id and version from the specified Cosmos DB table/container."""
    container = client.get_database_client(DB_NAME).get_container_client(SERVICE_DEFINATION_CONTAINER)
    query = f"SELECT * FROM c WHERE c.service_id = '{service_id}' AND c.version = '{version}'"
    items = list(container.query_items(query=query, enable_cross_partition_query=True))
    if not items:
        raise Exception(f"No record found for service ID '{service_id}' with version '{version}'.")
    for item in items:
        session_id = item['id']
        container.delete_item(item=session_id, partition_key=service_id)

# ---- Document Analytics ----
def get_all_document_metrics(service_id=None, version=None):
    container = client.get_database_client(DB_NAME).get_container_client(DOCUMENT_METRICS_CONTAINER)
    query = "SELECT * FROM c"
    if service_id and version:
        query += f" WHERE c.service_id = '{service_id}' AND c.version = '{version}'"

    items = list(container.query_items(query=query, enable_cross_partition_query=True))
    if not items:
        raise RuntimeError(f"No records found for service ID: {service_id}" + (f" and version: {version}" ) )
    return items

def get_document_metrics_by_name(document_name, service_id=None, version=None):
    container = client.get_database_client(DB_NAME).get_container_client(DOCUMENT_METRICS_CONTAINER)
    query = f"""
    SELECT * FROM c
    WHERE c.url LIKE '%{document_name}%'
    """
    if service_id and version:
        query += f" AND c.service_id = '{service_id}' AND c.version = '{version}'"
    return list(container.query_items(query=query, enable_cross_partition_query=True))

def get_document_metrics_by_date(start, end, service_id=None, version=None):
    container = client.get_database_client(DB_NAME).get_container_client(DOCUMENT_METRICS_CONTAINER)
    query = f"""
    SELECT * FROM c
    WHERE c.timestamp >= '{start.isoformat()}' AND c.timestamp <= '{end.isoformat()}'
    """
    if service_id and version:
        query += f" AND c.service_id = '{service_id}' AND c.version = '{version}'"
    return list(container.query_items(query=query, enable_cross_partition_query=True))

# ---- Chat session info ----
def get_all_chat_sessions(service_id=None, version=None):
    container = client.get_database_client(DB_NAME).get_container_client(CHAT_METRICS_CONTAINER)
    query = "SELECT * FROM c"
    if service_id and version:
        query += f" WHERE c.service_id = '{service_id}' AND c.version = '{version}'"
    return list(container.query_items(query=query, enable_cross_partition_query=True))

def get_chat_by_session_id(service_id=None, version=None):
    container = client.get_database_client(DB_NAME).get_container_client(CHAT_METRICS_CONTAINER)
    query = f"""
    SELECT * FROM c
    WHERE c.service_id = '{service_id}'
    """
    if service_id and version:
        query += f" AND c.service_id = '{service_id}' AND c.version = '{version}'"
    return list(container.query_items(query=query, enable_cross_partition_query=True))

def summarize_chat_sessions(items):
    service_response_times = defaultdict(list)
    feedback_counts = defaultdict(lambda: Counter({'good': 0, 'bad': 0, 'neutral': 0, 'none': 0}))
    session_query_count = Counter()

    for item in items:
        service = item.get("service_id", "unknown")
        session = item.get("session_id", "unknown")
        session_query_count[session] += 1

        rt = item.get("response_time")
        if isinstance(rt, (float, int)):
            service_response_times[service].append(rt)

        feedback = item.get("feedback", "").lower()
        if feedback not in ['good', 'bad', 'neutral']:
            feedback = 'none'
        feedback_counts[service][feedback] += 1

    avg_rt = {s: round(sum(times)/len(times), 2) for s, times in service_response_times.items()}
    avg_queries = round(sum(session_query_count.values()) / max(len(session_query_count), 1), 2)

    return {
        "average_response_time_by_service": avg_rt,
        "feedback_distribution_by_service": feedback_counts,
        "average_queries_per_session": avg_queries
    }

def get_document_count():
    container = client.get_database_client(DB_NAME).get_container_client(DOCUMENT_METRICS_CONTAINER)

    # Query all service_id values
    query = "SELECT c.service_id FROM c"
    items = list(container.query_items(query=query, enable_cross_partition_query=True))

    # Count how many times each service_id appears
    counts = defaultdict(int)
    for item in items:
        service_id = item.get("service_id", "unknown")
        counts[service_id] += 1

    return dict(counts)

def list_blobs(allowed_extensions = ('.pdf', '.docx', '.pptx', '.xlsx')):
    #random service selection
    service = services[random.randint(0, len(services))]
    service_id = service.get("service_id")
    storage = service.get("storage")
    connection_string = storage.get("connection_string")
    container_name = storage.get("blob_container")

    # Create the BlobServiceClient object
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container_name)

    # List all blobs in the container
    blob_list = container_client.list_blobs()
    file_paths = []

    for blob in blob_list:
        if blob.name.lower().endswith(allowed_extensions):
            file_paths.append(blob.name)

    return file_paths, service_id, connection_string, container_name

def get_chunks_from_doc(url, connection_string, container_name):
    file_obj = FileLoaderAzure(url.split("/")[-1], connection_string, container_name, url)()
    chunks = TextLoader.from_file(file_obj)

    return chunks

def load_chunk_driver( load_file_count = 2, load_chunk_count = 5):
    """
        load_file_count: works only with local_test = False, number of files to load
        load_chunk_count: works only with local_test = False number of random chunks to return
    """

    file_paths, service_id, connection_string, container_name = list_blobs()

    #random load_file_count indexes
    random_file_indexes = random.sample(range(len(file_paths)), load_file_count)
    chunks = []

    for i in random_file_indexes:
        url = file_paths[i]
        file_chunks = get_chunks_from_doc(url, connection_string, container_name)
        chunks.extend(file_chunks)

        #random load_chunk_count indexes
        random_chunk_indexes = random.sample(range(len(chunks)), load_chunk_count)
        return {"service_id": service_id, "chunks": [chunks[i] for i in random_chunk_indexes]}

def QnA(text: str) -> List[Tuple[str, str]]:
    """
    Generates multiple non-overlapping question-answer pairs from the entire given text.
    The whole text is treated as a single chunk.
    """
    text_chunks = [
        TextChunk(
            url="doc://full_text",
            text=text,
            document_id="doc_full",
            pages=[1],
            relevance=1.0
        )
    ]

    qa_pairs = []

    for chunk in text_chunks:
        # No knowledge base used
        questioner = Chatbot(FISLLM(), KnowledgeBases(), chat=Chat.from_memory())

        # System prompt with JSON format instruction
        questioner.chat.add('system', (
            "You are a question generator and answerer. Your task is to create multiple comprehensive, non-overlapping "
            "question-answer pairs that together cover the entire text chunk below. Avoid yes/no questions or one-line answers. "
            "Use only the information from the text chunk. Each answer should be distinct and should contain data only from the text. "
            "Return the output strictly in the following JSON format:\n\n"
            "{\n"
            "  \"qa_pairs\": [\n"
            "    {\"question\": \"<question1>\", \"answer\": \"<answer1>\"},\n"
            "    {\"question\": \"<question2>\", \"answer\": \"<answer2>\"}\n"
            "  ]\n"
            "}"
        ))

        # Prompt with full text chunk
        prompt = f"Generate multiple question-answer pairs that together cover the following text without overlapping:\n\n{chunk.text}"
        qa_response = questioner(prompt)

        # Parse JSON response
        try:
            qa_data = json.loads(qa_response['answer'])
            for pair in qa_data.get("qa_pairs", []):
                q = pair.get("question", "").strip()
                a = pair.get("answer", "").strip()
                if len(a.split()) > 4:
                    qa_pairs.append((f"Question: {q}", f"Answer: {a}"))
        except json.JSONDecodeError as e:
            logger.error("Failed to parse JSON:", e)
      

        questioner.chat.clear()

    return qa_pairs

def fisbot_answers(qa_pairs: list, service_id: str):
    questions = []
    baseline_answers = []
    bot_answers = {}
    versions = []

    for item in qa_pairs:
        questions.append(item[0].split(":")[1])
        baseline_answers.append(item[1].split(":")[1])

    llm = FISLLM()
    service_id = service_id

    for item in services:
        if(item['service_id'] == service_id):
            versions.append(item['version'])
            
    #creating knowledge bases as per service_id and version.
    for num in range(len(versions)):
        version = versions[num]
        kb = create_knowledge_base(FISLLM(), service_id, version)
        bot = Chatbot.assistant(llm, kb, Chat.from_memory(),service_id)
        for ques in range(len(questions)):
            bot_answer = bot(questions[ques])
            bot_answers[f"version_{versions[num]}_Ques_{ques+1}"] = bot_answer['answer']

    return questions, baseline_answers, bot_answers, service_id

def add_record_document_qna_container(session_id: str, service_id: str, question: str, chatbot_answer: str, expected_baseline_answer: str, table_name: str, version: str, response_source: str):
    """Adds the records to the specified Cosmos DB table/container."""
    '''expected_baseline_answer: Human annotated answer / bot generated ans without knowledge base'''
    '''response_source: `Ai` '''

    time_stamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    data = {"service_id": service_id, "id": session_id, "question": question, "chatbot_answer": chatbot_answer, "expected_baseline_answer": expected_baseline_answer,
            "version": version, "time_stamp": time_stamp, "response_source": response_source, "metrics": ""}

    CosmosDBTable(CONFIG['DB_URL'], CONFIG['DB_KEY'], CONFIG['DB_NAME'], table_name).add(data)


def prepare_qna_with_llm():
    qa_pairs = []
  
    # get the data from blob and make chunks
    try:
        data = load_chunk_driver()
        service_id = data['service_id']
        chunks = data['chunks']
    except Exception as e:
        raise RuntimeError(f"Error in loading chunks: {e}")
    
    # QnA pair generation
    try:
        for item in chunks:
            chunk = item.text
            qa_pair = QnA(chunk)
            for qna in qa_pair:
                qa_pairs.append(qna)
    except Exception as e:
        raise RuntimeError(f"Error in generating QnA pairs: {e}")
    
    # chatbot answers generation
    try:
        questions, baseline_answers, bot_answers, service_id = fisbot_answers(qa_pairs, service_id)
    except Exception as e:
        raise RuntimeError(f"Error in getting bot answers: {e}")

    # add data to container
    try:
      for i in range(len(bot_answers)):
        key = list(bot_answers.keys())[i]
        version = key.split("_")[1]
        bot_ans = bot_answers[key]
        
        time_stamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        chat_bot_ans =[{ 'timestamp' : time_stamp, 'answer' : bot_ans }]
        
        question_index = i % len(questions)  # Cycle through questions
        add_record_document_qna_container(
          str(uuid.uuid4()), service_id, questions[question_index], chat_bot_ans,
          baseline_answers[question_index], DOCUMENT_QNA_AI_CONTAINER,
          version, "Ai")
      
      return len(bot_answers)
    except Exception as e:
      raise RuntimeError(f"Error in adding records to container: {e}")

def get_items_no_metrics():
    try:
        container = client.get_database_client(DB_NAME).get_container_client(DOCUMENT_QNA_AI_CONTAINER )
        query = f"SELECT * FROM c WHERE c.metrics = ''" # getting items where metrics is null
        items = list(container.query_items(query=query, enable_cross_partition_query=True))
    except Exception as e:
        raise RuntimeError(f"Issues with fetching QnA items or return items length is zero. {e}")
    return items

def get_metrics(true, pred):
    
    tokenizer = FISLLM().tokenizer
    return Metrics(tokenizer)(true, pred)

def update_item_to_azure_container(item):
    try:
        container = client.get_database_client(DB_NAME).get_container_client(DOCUMENT_QNA_AI_CONTAINER )
        container.replace_item(item=item['id'], body=item)
    except Exception as e:
        raise RuntimeError(f"Issues with updating items to Azure container {e}")

def calculate_metrics(): 
    items = get_items_no_metrics()
    for item in items:
      metric = get_metrics(item['expected_baseline_answer'], item['chatbot_answer'][0]['answer'])

      time_stamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
      item['metrics'] =[{ 'timestamp' : time_stamp, 'metric' : metric }]
      
      update_item_to_azure_container(item)

    return len(items)

def get_service_blob_container(service_id, version):
  for service in services:
    if service['service_id']==service_id and service['version']==version:
      return FileLoaderAzure.to_public_url(CONFIG['BLOB_URL'], container=service.get('storage',{}).get('blob_container',''))
  raise Exception(f'Service definition not found for service_id={service_id} and version={version}.')

def view_qna_records_by_service_id(service_id: str, response_source: str):
    """Returns all records from the specified Cosmos DB table/container based on service_id, response_source(Human/Ai)"""
    try:
      if response_source == 'Human':
        container = client.get_database_client(DB_NAME).get_container_client(DOCUMENT_QNA_CONTAINER)
      else:   #'Ai'
        container = client.get_database_client(DB_NAME).get_container_client(DOCUMENT_QNA_AI_CONTAINER)
      
      query = f"SELECT * FROM c WHERE c.service_id = '{service_id}' AND c.response_source = '{response_source}'"

      items = list(container.query_items(query=query, enable_cross_partition_query=True))
      if not items:
          return []
      return items
    except Exception as e:
      raise RuntimeError(f"Issues while getting records from the Azure Container: {service_id}, {response_source}")


#This is for future use only.
def add_human_records_document_qna_container(data: list):
  """Adds the records to the specified Cosmos Db table/container"""
  try:
      for record in data:
          time_stamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
          data_record = {"service_id": record['service_id'], "id": str(uuid.uuid4()), "question": record['question'], 
                  "chatbot_answer": [], "expected_baseline_answer": record['expected_baseline_answer'], "time_stamp" : time_stamp,
                  "version": record['version'], "response_source": "Human", "metrics": [], "file_source": record['source_file_name']}
          
          CosmosDBTable(CONFIG['DB_URL'], CONFIG['DB_KEY'], CONFIG['DB_NAME'], DOCUMENT_QNA_CONTAINER).add(data_record)

  except Exception as e:
      raise RuntimeError(f"Issues while Adding record to the Azure Container")
