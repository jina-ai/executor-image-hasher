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
    """
    An executer to encode the images using the `comparable` hashing techniques
    """

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
        """
        :param hash_type: the hashing technique used to encode the images. By default, set to `perceptual`.
        Possible values are: `perceptual`, `average`, `difference`, and `wavelet`.
        Implementation of the algorithm can be found here - https://github.com/JohannesBuchner/imagehash
        :param hash_size: the size of the encoded hash value. Should not be less than `2`
        :param average_hash_args: the arguments for the `average` hash encoding type. By default `avg_pixels`
        set to mean. The other possible value for `avg_pixels is `median`
        For example: `average_hash_args={'avg_pixels': 'median'}`
        :param perceptual_hash_args: the arguments for the `perceptual` hash encoding type. This dict should contain
        `highfreq_factor` as the only key. The default value set is 4.
        For example: `perceptual_hash_args={'highfreq_factors': 4}`
        :param wavelet_hash_args: the arguments for `wavelet` hash encoding type. By default `image_scale` is equal
         to max power of 2 for an input image and the `mode` is 'haar', which is Haar wavelets. The other possible
         value for `mode` is db4 - Daubechies wavelets.
         :param is_embed_bool: Set to `True` to encode the images into boolean embeddings using the hashing technique.
         By default set to `False` to encode the images as `np.uint8` embeddings values.
         :param traversal_paths: The default traversal path on docs, e.g. ['r'], ['c']
        """
        super().__init__(**kwargs)
        if not hash_type or hash_type not in HASH_TYPE:
            raise ValueError(
                f'Please select one of the available `hash_type`: {HASH_TYPE}'
            )
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
        """
        Read the numpy arrays from the Document.blob, and encode it into embeddings using the
        comparable hashing technique. Features in the image are used to generate a distinct
        (but not unique) fingerprint, and these fingerprints are comparable.

        :param docs: A document array with documents to create embeddings for. Only the
            documents that have the ``blob`` attribute will get embeddings. The ``blob``
            attribute should be the numpy array of the image, and should have dtype
            ``np.uint8``
        :param parameters: A dictionary that contains parameters to control encoding.
            The accepted keys are: `hash_type`, `hash_size`, `perceptual_hash_args`,
            `average_hash_args`, `wavelet_hash_args`, and `traversal_paths`.
            For example: `parameters={'hash_type': 'average', 'average_hash_args': {'avg_pixels': 'median'}}`
        """
        missing_blob = []
        hash_type = parameters.get('hash_type', self.hash_type)
        hash_size = parameters.get('hash_size', self.hash_size)

        for doc in docs.traverse_flat(
            parameters.get('traversal_paths', self.traversal_paths)
        ):
            if doc.blob is None:
                missing_blob.append(doc.id)
                continue

            image = Image.fromarray(doc.blob)
            hash_hex = None
            try:
                if hash_type == 'wavelet':
                    wavelet_hash_args = deepcopy(self._wavelet_hash_args)
                    wavelet_hash_args.update(parameters.get('wavelet_hash_args', {}))
                    hash_hex = imagehash.whash(
                        image, hash_size=hash_size, **wavelet_hash_args
                    )
                elif hash_type == 'average':
                    average_hash_args = deepcopy(self._average_hash_args)
                    average_hash_args.update(parameters.get('average_hash_args', {}))
                    hash_hex = imagehash.average_hash(
                        image, hash_size=hash_size, **average_hash_args
                    )
                elif hash_type == 'perceptual':
                    perceptual_hash_args = deepcopy(self._perceptual_hash_args)
                    perceptual_hash_args.update(
                        parameters.get('perceptual_hash_args', {})
                    )
                    hash_hex = imagehash.phash(
                        image, hash_size=hash_size, **perceptual_hash_args
                    )
                else:
                    hash_hex = imagehash.phash(image, hash_size=hash_size)
            except AssertionError as e:
                self.logger.error(f'Image hashing failed, {e}')

            if hash_hex is not None:
                if self.is_embed_bool:
                    doc.embedding = np.squeeze(
                        np.array(hash_hex.hash).reshape(1, hash_size ** 2)
                    )
                else:
                    doc.embedding = np.squeeze(
                        np.packbits(
                            np.array(hash_hex.hash).reshape(1, hash_size ** 2),
                            axis=1,
                        )
                    )
            else:
                self.logger.warning(f'Could not set embeddings for {doc.id}')

        if len(missing_blob) > 0:
            self.logger.error(
                f'No blob passed for the following Documents with ids: {missing_blob}'
            )
