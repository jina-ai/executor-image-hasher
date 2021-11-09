__copyright__ = 'Copyright (c) 2020-2021 Jina AI Limited. All rights reserved.'
__license__ = 'Apache-2.0'

from copy import deepcopy
from typing import Iterable, Dict, Optional

import imagehash
import numpy as np
from PIL import Image
from jina import Executor, DocumentArray, requests
from jina.logging.logger import JinaLogger

HASH_TYPE = ['perceptual', 'average', 'difference', 'wavelet']


class ImageHasher(Executor):
    def __init__(
        self,
        hash_type: str = 'perceptual',
        hash_size: int = 8,
        average_hash_args: Optional[Dict] = None,
        perceptual_hash_args: Optional[Dict] = None,
        wavelet_hash_args: Optional[Dict] = None,
        is_embed_bool: bool = False,
        traversal_paths: Iterable[str] = ('r',),
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not hash_type or hash_type not in HASH_TYPE:
            raise ValueError('Please select one of the `hash_type`')
        self.hash_type = hash_type
        self.hash_size = hash_size

        self._average_hash_args = average_hash_args or {}
        if 'avg_pixels' in self._average_hash_args:
            if self._average_hash_args['avg_pixels'] == 'mean':
                self._average_hash_args['mean'] = np.mean
            else:
                self._average_hash_args['mean'] = np.median
            del self._average_hash_args['avg_pixels']

        self._perceptual_hash_args = perceptual_hash_args or {}
        self._perceptual_hash_args.setdefault('highfreq_factor', 4)

        self._wavelet_hash_args = wavelet_hash_args or {}
        self._wavelet_hash_args.setdefault('image_scale', None)
        self._wavelet_hash_args.setdefault('mode', 'haar')

        self.is_embed_bool = is_embed_bool
        self.traversal_paths = traversal_paths
        self.logger = JinaLogger(
            getattr(self.metas, 'name', self.__class__.__name__)
        ).logger

    @requests
    def encode(self, docs: DocumentArray, parameters: Dict = {}, **kwargs):
        for doc in docs.traverse_flat(
            parameters.get('traversal_paths', self.traversal_paths)
        ):
            if doc.blob is None:
                self.logger.error(f'No blob passed for the Document with id: {doc.id}')
                continue

            image = Image.fromarray(doc.blob)
            hash_hex = None
            try:
                if self.hash_type == 'wavelet':
                    wavelet_hash_args = deepcopy(self._wavelet_hash_args)
                    wavelet_hash_args.update(parameters.get('wavelet_hash_args', {}))
                    hash_hex = imagehash.whash(
                        image, hash_size=self.hash_size, **wavelet_hash_args
                    )
                elif self.hash_type == 'average':
                    average_hash_args = deepcopy(self._average_hash_args)
                    average_hash_args.update(parameters.get('average_hash_args', {}))
                    hash_hex = imagehash.average_hash(
                        image, hash_size=self.hash_size, **average_hash_args
                    )
                elif self.hash_type == 'perceptual':
                    perceptual_hash_args = deepcopy(self._perceptual_hash_args)
                    perceptual_hash_args.update(
                        parameters.get('perceptual_hash_args', {})
                    )
                    hash_hex = imagehash.phash(
                        image, hash_size=self.hash_size, **perceptual_hash_args
                    )
                else:
                    hash_hex = imagehash.phash(image, hash_size=self.hash_size)
            except AssertionError as e:
                self.logger.error(f'Image hashing failed, {e}')

            if hash_hex is not None:
                if self.is_embed_bool:
                    doc.embedding = np.squeeze(
                        np.array(hash_hex.hash).reshape(1, self.hash_size ** 2)
                    )
                else:
                    doc.embedding = np.squeeze(
                        np.packbits(
                            np.array(hash_hex.hash).reshape(1, self.hash_size ** 2),
                            axis=1,
                        )
                    )
            else:
                self.logger.warning(f'Could not set embeddings for {doc.id}')
