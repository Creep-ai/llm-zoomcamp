from typing import Any, Optional

import numpy as np
import pandas as pd
import typing_extensions
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class Index:
    """A simple search index using TF-IDF and cosine similarity for text fields and exact matching for keyword fields.

    Attributes
    ----------
        text_fields (list): List of text field names to index.
        keyword_fields (list): List of keyword field names to index.
        vectorizers (dict): Dictionary of TfidfVectorizer instances for each text field.
        keyword_df (pd.DataFrame): DataFrame containing keyword field data.
        text_matrices (dict): Dictionary of TF-IDF matrices for each text field.
        docs (list): List of documents indexed.

    """

    def __init__(
        self, text_fields: list[str], keyword_fields: dict[str, str], vectorizer_params: Optional[dict[str, Any]] = None
    ) -> None:
        """Initialize the Index with specified text and keyword fields.

        Args:
        ----
            text_fields (list): List of text field names to index.
            keyword_fields (list): List of keyword field names to index.
            vectorizer_params (dict): Optional parameters to pass to TfidfVectorizer.

        """
        if vectorizer_params is None:
            vectorizer_params = {}
        self.text_fields = text_fields
        self.keyword_fields = keyword_fields

        self.vectorizers = {field: TfidfVectorizer(**vectorizer_params) for field in text_fields}
        self.keyword_df: pd.DataFrame | None = None
        self.text_matrices: dict[str, str] = {}
        self.docs: list[dict[str, str]] = []

    def fit(self, docs: list[dict[str, str]]) -> typing_extensions.Self:
        """Fits the index with the provided documents.

        Args:
        ----
            docs (list of dict): List of documents to index. Each document is a dictionary.

        """
        self.docs = docs
        keyword_data: dict[str, list[str]] = {field: [] for field in self.keyword_fields}

        for field in self.text_fields:
            texts = [doc.get(field, "") for doc in docs]
            self.text_matrices[field] = self.vectorizers[field].fit_transform(texts)

        for doc in docs:
            for field in self.keyword_fields:
                keyword_data[field].append(doc.get(field, ""))

        self.keyword_df = pd.DataFrame(keyword_data)

        return self

    def search(
        self,
        query: str,
        filter_dict: Optional[dict[str, str]] = None,
        boost_dict: Optional[dict[str, float]] = None,
        num_results: int = 10,
    ) -> list[dict[str, str]]:
        """Search the index with the given query, filters, and boost parameters.

        Args:
        ----
            query (str): The search query string.
            filter_dict (dict): Dictionary of keyword fields to filter by.
                                Keys are field names and values are the values to filter by.
            boost_dict (dict): Dictionary of boost scores for text fields.
                                Keys are field names and values are the boost scores.
            num_results (int): The number of top results to return. Defaults to 10.

        Returns:
        -------
            list of dict: List of documents matching the search criteria, ranked by relevance.

        """
        if boost_dict is None:
            boost_dict = {}
        if filter_dict is None:
            filter_dict = {}
        query_vecs = {field: self.vectorizers[field].transform([query]) for field in self.text_fields}
        scores = np.zeros(len(self.docs))

        # Compute cosine similarity for each text field and apply boost
        for field, query_vec in query_vecs.items():
            sim = cosine_similarity(query_vec, self.text_matrices[field]).flatten()
            boost = boost_dict.get(field, 1)
            scores += sim * boost

        # Apply keyword filters
        for field, value in filter_dict.items():
            if field in self.keyword_fields:
                if self.keyword_df is not None:
                    mask = self.keyword_df[field] == value
                else:
                    raise ValueError("self.keywords should be dataframe, got None")
                scores = scores * mask.to_numpy()

        # Use argpartition to get top num_results indices
        top_indices = np.argpartition(scores, -num_results)[-num_results:]
        top_indices = top_indices[np.argsort(-scores[top_indices])]

        # Filter out zero-score results
        return [self.docs[i] for i in top_indices if scores[i] > 0]
