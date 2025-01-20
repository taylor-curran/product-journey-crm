import asyncio
from datetime import timedelta
from typing import List, Literal, Optional

from prefect import flow, task
from prefect.tasks import task_input_hash
from raggy.documents import Document
from raggy.loaders.base import Loader
from raggy.loaders.github import GitHubRepoLoader
from raggy.loaders.web import SitemapLoader
from raggy.vectorstores.tpuf import TurboPuffer


@task(
    retries=2,
    retry_delay_seconds=[3, 60],
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(days=1),
    task_run_name="Run {loader.__class__.__name__}",
    persist_result=True,
)
async def run_loader(loader: Loader) -> List[Document]:
    return await loader.load()


@task(cache_key_fn=task_input_hash, cache_expiration=timedelta(days=1))
async def add_documents(
    tpuf: TurboPuffer,
    documents: list[Document],
    mode: Literal["upsert", "reset"],
    batch_size: int = 100,
    max_concurrent: int = 8,
) -> None:
    """Add documents to TurboPuffer with batching support"""
    if mode == "reset":
        # TurboPuffer doesn't have a direct reset, but we can create a new namespace
        await tpuf.delete_namespace()
        await tpuf.upsert_batched(
            documents=documents, batch_size=batch_size, max_concurrent=max_concurrent
        )
    else:
        await tpuf.upsert_batched(
            documents=documents, batch_size=batch_size, max_concurrent=max_concurrent
        )


@flow(name="Update Knowledge", log_prints=True)
async def refresh_tpuf(
    # Vectorstore params
    namespace: str = "default",
    mode: Literal["upsert", "reset"] = "upsert",
    # Optional sitemap params
    sitemap_urls: Optional[List[str]] = None,
    sitemap_exclude: Optional[List[str]] = None,
    # Optional GitHub params
    github_repo: Optional[str] = None,
    github_include_globs: Optional[List[str]] = None,
    # Batch processing params
    batch_size: int = 100,
    max_concurrent: int = 8,
):
    """
    Flow updating the TurboPuffer vectorstore with documents from one or more data sources:
     - Zero or more sitemaps (SitemapLoader)
     - Optionally a GitHub repo (GitHubRepoLoader)
    """
    loaders = []

    # Add the sitemap loader if URLs are provided
    if sitemap_urls:
        loaders.append(
            SitemapLoader(
                urls=sitemap_urls,
                exclude=sitemap_exclude or [],
            )
        )

    # Add the GitHub loader if a repo is provided
    if github_repo:
        loaders.append(
            GitHubRepoLoader(
                repo=github_repo,
                include_globs=github_include_globs or ["README.md"],
            )
        )

    # If no loaders were provided, you might want to raise an error or skip
    if not loaders:
        print("No loaders specified — nothing to do.")
        return

    # Orchestrate loading of documents
    documents: List[Document] = [
        doc
        for future in run_loader.map(loaders)  # type: ignore
        for doc in future.result()  # type: ignore
    ]

    print(f"Loaded {len(documents)} documents from specified sources.")

    with TurboPuffer(namespace=namespace) as tpuf:
        await add_documents(
            tpuf=tpuf,
            documents=documents,
            mode=mode,
            batch_size=batch_size,
            max_concurrent=max_concurrent,
        )
        print(f"Added {len(documents)} documents to the {namespace} namespace.")


if __name__ == "__main__":
    # Example usage with Prefect data
    asyncio.run(
        refresh_tpuf(
            namespace="test-tay",
            mode="upsert",
            sitemap_urls=[
                "https://docs.prefect.io/sitemap.xml",
                "https://prefect.io/sitemap.xml",
            ],
            sitemap_exclude=["api-ref", "www.prefect.io/events"],
            github_repo="PrefectHQ/prefect",
            github_include_globs=["README.md"],
        )
    )

# reset is for testing
# upsert is for production
# upsert is for production
