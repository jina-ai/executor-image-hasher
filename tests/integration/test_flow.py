__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import numpy as np
import pytest
from jina import Document, DocumentArray, Flow

from executor import ImageHasher


@pytest.mark.parametrize("request_size", [1, 10, 50, 100])
def test_integration(request_size: int):
    docs = DocumentArray(
        [
            Document(blob=np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
            for _ in range(50)
        ]
    )
    with Flow(return_results=True).add(uses=ImageHasher) as flow:
        resp = flow.post(
            on="/encode",
            inputs=docs,
            request_size=request_size,
            return_results=True,
        )

    assert sum(len(resp_batch.docs) for resp_batch in resp) == 50
    for r in resp:
        for doc in r.docs:
            assert doc.embedding is not None
            assert doc.embedding.shape == (8,)
