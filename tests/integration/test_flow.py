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
            Document(tensor=np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
            for _ in range(50)
        ]
    )
    with Flow().add(uses=ImageHasher) as flow:
        da = flow.post(
            on="/encode",
            inputs=docs,
            request_size=request_size
        )

    assert len(da) == 50
    for doc in da:
        assert doc.embedding is not None
        assert doc.embedding.shape == (8,)
