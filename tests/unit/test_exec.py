__copyright__ = 'Copyright (c) 2020-2021 Jina AI Limited. All rights reserved.'
__license__ = 'Apache-2.0'

import os
from pathlib import Path

import numpy as np
import pytest
from jina import Document, DocumentArray, Executor

from executor import ImageHasher

cur_dir = os.path.dirname(os.path.abspath(__file__))


def test_config():
    ex = Executor.load_config(str(Path(__file__).parents[2] / 'config.yml'))
    assert ex.hash_type == 'perceptual'
    assert ex.hash_size == 8


def test_no_documents(hasher: ImageHasher):
    docs = DocumentArray()
    hasher.encode(docs=docs)
    assert len(docs) == 0  # SUCCESS


def test_docs_no_blobs(hasher: ImageHasher):
    docs = DocumentArray([Document()])
    hasher.encode(docs=docs)
    assert len(docs) == 1
    assert len(docs[0].chunks) == 0


def test_embed_type(hasher, docs):
    hasher.encode(docs=docs)
    assert docs.embeddings.dtype == np.uint8

    hasher_bool = ImageHasher(is_embed_bool=True)
    hasher_bool.encode(docs=docs)
    assert docs.embeddings.dtype == np.bool


def test_contrast_match(hasher, docs_contrast):
    hasher.encode(docs=docs_contrast)
    docs_contrast.match(docs_contrast, metric='euclidean', use_scipy=True)
    for i, doc in enumerate(docs_contrast):
        assert (
            doc.matches[0].scores['euclidean'].value
            == doc.matches[1].scores['euclidean'].value
        )


def test_match_quality(hasher: ImageHasher, docs):
    hasher.encode(docs=docs)
    docs.match(docs, metric='euclidean', use_scipy=True)
    matches = ['kids2', 'kids1', 'paprika2', 'paprika1']
    for i, doc in enumerate(docs):
        assert doc.matches[1].id == matches[i]


@pytest.mark.parametrize(
    'hash_type', ['perceptual', 'average', 'wavelet', 'difference']
)
@pytest.mark.parametrize('hash_size', [8, 16])
def test_hashing_technique(hash_type, hash_size, docs):
    hasher = ImageHasher(hash_type=hash_type, hash_size=hash_size)
    hasher.encode(docs=docs)
    for doc in docs:
        assert doc.embedding is not None
