from datetime import timedelta
from typing import Literal, Optional, List

from chromadb.api.models.Collection import Document as ChromaDocument
from prefect import flow, task
from prefect.tasks import task_input_hash

from raggy.documents import Document
from raggy.loaders.base import Loader
from raggy.loaders.github import GitHubRepoLoader
from raggy.loaders.web import SitemapLoader
from raggy.vectorstores.chroma import Chroma, ChromaClientType


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


@task
def add_documents(
    chroma: Chroma, documents: list[Document], mode: Literal["upsert", "reset"]
) -> list[ChromaDocument]:
    if mode == "reset":
        chroma.reset_collection()
        docs = chroma.add(documents)
    elif mode == "upsert":
        docs = chroma.upsert(documents)
    return docs


@flow(name="Update Knowledge", log_prints=True)
def refresh_chroma(
    # Vectorstore params
    collection_name: str = "default",
    chroma_client_type: ChromaClientType = "base",
    mode: Literal["upsert", "reset"] = "upsert",
    # Optional sitemap params
    sitemap_urls: Optional[List[str]] = None,
    sitemap_exclude: Optional[List[str]] = None,
    # Optional GitHub params
    github_repo: Optional[str] = None,
    github_include_globs: Optional[List[str]] = None,
):
    """
    Flow updating the vectorstore with documents from one or more data sources:
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

    with Chroma(
        collection_name=collection_name, client_type=chroma_client_type
    ) as chroma:
        docs = add_documents(chroma, documents, mode)

        print(f"Added {len(docs)} documents to the {collection_name} collection.")  # type: ignore


if __name__ == "__main__":
    # Example usage with Prefect data
    refresh_chroma(
        collection_name="test",
        chroma_client_type="base",
        mode="reset",
        sitemap_urls=[
            "https://docs.prefect.io/sitemap.xml",
            "https://prefect.io/sitemap.xml",
        ],
        sitemap_exclude=["api-ref", "www.prefect.io/events"],
        github_repo="PrefectHQ/prefect",
        github_include_globs=["README.md"],
    )
