def remove(self, url:str) -> bool:
  client = SearchClient(endpoint=self.url, index_name=self.index, credential=self.credential)
  documents = client.search(search_text=f'url:{url}')
  actions = [{'@search.action':'delete','id': doc['id']} for doc in documents if json.loads(doc.get('metadata','')).get('url')==url]
  if not actions:
    return False
  removed = client.upload_documents(documents=actions)
  return bool(removed)



def create_knowledge_base(llm:LLM, service_id:str, version:str) -> KnowledgeBase:
  """
  Creates vector database as a knowledge base.

  :param llm: Underlying LLM with embeddings for text vectorization.
  :param service_id: Service ID, which is used as the first part of index name.
  :param version: Service version, which is used as the second part of index name.
  """
  return CognitiveSearch(url=CONFIG['VDB_URL'], key=CONFIG['VDB_KEY'], index=f'{service_id}_{version}', llm=llm)


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
            st = time.time()
            kb.remove(missing_url)
            logger.info(f'Time to remove one file:{str(time.time()-st)}')
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
