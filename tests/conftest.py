from pathlib import Path

import pytest
import numpy as np
from PIL import Image
from jina import Document, DocumentArray

from executor import ImageHasher

data_dir = (Path(__file__).parent / 'toy_data').absolute()


@pytest.fixture(scope='function')
def hasher():
    return ImageHasher()


@pytest.fixture(scope='function')
def docs():
    return DocumentArray(
        [
            Document(id='kids1', blob=np.array(Image.open((data_dir / 'kids1.jpg')))),
            Document(id='kids2', blob=np.array(Image.open((data_dir / 'kids2.jpg')))),
            Document(
                id='paprika1', blob=np.array(Image.open((data_dir / 'paprika1.png')))
            ),
            Document(
                id='paprika2', blob=np.array(Image.open((data_dir / 'paprika2.png')))
            ),
        ]
    )


@pytest.fixture(scope='function')
def docs_contrast():
    return DocumentArray(
        [
            Document(id='ct1', blob=np.array(Image.open((data_dir / 'scene1.png')))),
            Document(id='ct2', blob=np.array(Image.open((data_dir / 'scene2.png')))),
        ]
    )
