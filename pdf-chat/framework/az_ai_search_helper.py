import json
import os
import openai
from pathlib import Path
import dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient, SearchIndexingBufferedSender
from azure.search.documents.indexes import SearchIndexClient
from langchain.vectorstores.azuresearch import AzureSearch
from langchain.embeddings import AzureOpenAIEmbeddings
from azure.search.documents.indexes.models import (
    FreshnessScoringFunction,
    FreshnessScoringParameters,
    ScoringProfile,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    TextWeights,
)

env_name = os.environ["APP_ENV"] if "APP_ENV" in os.environ else "local"

# Load env settings
env_file_path = Path(f"./.env.{env_name}")
print(f"Loading environment from: {env_file_path}")
with open(env_file_path) as f:
    dotenv.load_dotenv(dotenv_path=env_file_path)
# print(os.environ)


def get_az_search_client():
    endpoint = os.environ["AZURE_SEARCH_ENDPOINT"]
    index_name = os.environ["AZURE_SEARCH_INDEX_NAME"]
    key = os.environ["AZURE_SEARCH_ADMIN_KEY"]

    credential = AzureKeyCredential(key)
    client = SearchClient(endpoint=endpoint,
                          index_name=index_name,
                          credential=credential)
    return client


def get_az_search_indices():

    # Get all the indices from the Azure Search service
    return ["azure-plat-services-vector-search", "langchain-vector-demo"]


def get_index_fields(index_name):
    if index_name == "azure-plat-services-vector-search":
        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String,
                        key=True, sortable=True, filterable=True, facetable=True),
            SearchableField(name="title", type=SearchFieldDataType.String),
            SearchableField(name="content", type=SearchFieldDataType.String),
            SearchableField(name="category", type=SearchFieldDataType.String,
                            filterable=True),
            SearchField(name="title_vector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                        searchable=True, vector_search_dimensions=1536, vector_search_configuration="myHnswProfile"),
            SearchField(name="content_vector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                        searchable=True, vector_search_dimensions=1536, vector_search_configuration="myHnswProfile"),
        ]

    else:
        fields = [
            SimpleField(
                name="id",
                type=SearchFieldDataType.String,
                key=True,
                filterable=True,
            ),
            SearchableField(
                name="content",
                type=SearchFieldDataType.String,
                searchable=True,
            ),
            SearchField(
                name="content_vector",
                type=SearchFieldDataType.Collection(
                    SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=len(embedding_function("Text")),
                vector_search_configuration="default",
            ),
            SearchableField(
                name="metadata",
                type=SearchFieldDataType.String,
                searchable=True,
            )
        ]

    return fields
