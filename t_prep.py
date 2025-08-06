
from typing import List, Union
from azure.search.documents import SearchClient

def remove(self, urls: Union[str, List[str]], batch_size: int = 1000) -> bool:
    """
    Delete one or many documents by `url` in *batched* calls so the whole
    purge finishes in minutes, not hours.
    """
    # normalise input
    if isinstance(urls, str):
        urls = [urls]
    if not urls:
        return False

    # re-use a single SearchClient (created on first use)
    if not hasattr(self, "_client"):
        self._client = SearchClient(
            endpoint=self.url,
            index_name=self.index,
            credential=self.credential,
        )
    client = self._client

    removed_any = False
    for i in range(0, len(urls), batch_size):
        batch = urls[i : i + batch_size]

        # one query returns ids for the whole batch
        filter_expr = f"search.in(url, '{','.join(batch)}', ',')"
        docs = client.search(
            search_text="*",
            filter=filter_expr,
            select="id",
            top=batch_size,
        )
        ids = [d["id"] for d in docs]

        if not ids:
            continue

        # one upload_documents call deletes the chunk
        actions = [{"@search.action": "delete", "id": _id} for _id in ids]
        client.upload_documents(actions)
        removed_any = True

    return removed_any


def prepare():
    logger.info(f" {service_id} ({version}) Preparing knowledge base...")
    try:
        # 1. build / reuse KB
        kb = create_knowledge_base(FISLLM(), service_id, version)

        # 2. build loaders (unchanged)
        skip = create_skip_function(kb)
        file_loaders = create_aws_loaders(skip) + create_azure_loaders(skip)

        # 3. log unsupported files (unchanged) …
        if unsupported_urls:
            unsupported_ext2count = {}
            for u in unsupported_urls:
                ext = FileObject(u).ext
                unsupported_ext2count[ext] = unsupported_ext2count.get(ext, 0) + 1
            cnt_str = ", ".join(f"{c} x {e}" for e, c in unsupported_ext2count.items())
            logger.warning(f"⚠ {service_id} ({version}) Skipping {len(unsupported_urls)} unsupported files: {cnt_str}")

        # ------------------------------------------------------------------ #
        # fast, batched delete of all missing URLs in one call     #
        # ------------------------------------------------------------------ #
        if missing_urls:
            missing_to_remove = [
                u for u in missing_urls
                if not confluence_url or not u.startswith(confluence_url)
            ]
            if missing_to_remove:
                logger.info(f"🔄 {service_id} ({version}) Removing {len(missing_to_remove)} missing files…")
                t0 = time.time()
                kb.remove(missing_to_remove)                 # batched delete
                logger.info(f" Removed in {round(time.time() - t0, 2)} s")

        # 4. short-circuit if nothing new to add
        if not file_loaders and not confluence_space:
            logger.info(f" {service_id} ({version}) is up to date [no new files].")
            return True

        # 5. ingest new / updated files (unchanged) …
        logger.info(f"🔄 {service_id} ({version}) Processing {len(supported_urls)} new files: {', '.join(supported_urls)}")
        update_results = []
        start_time = time.time()
        for loader in tqdm(file_loaders):
            update_results.append(update_knowledge_base(kb, loader=loader))
        if confluence_space:
            update_results.append(update_knowledge_base(kb, confluence_file=True))
        update_seconds = round(time.time() - start_time, 4)

        return log(update_results, update_seconds) and test_chatbot(kb)

    except Exception as e:
        logger.error(f"🚩 {service_id} ({version}) Failed to prepare knowledge base: {e}", exc_info=True)
        return False
