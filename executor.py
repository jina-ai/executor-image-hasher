__copyright__ = 'Copyright (c) 2020-2021 Jina AI Limited. All rights reserved.'
__license__ = 'Apache-2.0'

from copy import deepcopy
from typing import Iterable, Dict, Optional

import imagehash
import numpy as np
from PIL import Image
from jina import Executor, DocumentArray, requests
from jina.logging.logger import JinaLogger

import warnings

HASH_TYPE = ['phash', 'average_hash', 'dhash', 'whash']


class ImageHasher(Executor):
    """
    An executer to encode the images using the `comparable` hashing techniques
    """

    def __init__(
        self,
        hash_type: str = 'phash',
        hash_size: int = 8,
        hash_func_args: Optional[Dict] = None,
        is_embed_bool: bool = False,
        access_paths: str = '@r',
        traversal_paths: Optional[str] = None,
        **kwargs,
    ):
        """
        :param hash_type: the hashing technique used to encode the images. By default, set to `phash`.
        Possible values are: `phash`, `average_hash`, `dhash`, and `whash`.
        Implementation of the algorithm can be found here - https://github.com/JohannesBuchner/imagehash
        :param hash_size: the size of the encoded hash value. Should not be less than `2`
        :param hash_func_args: the arguments for the hashing functions - `phash` and `whash`.
         - This dict should contain `highfreq_factor` as the only key for `phash`. The default value set is 4.
         - This dict should contain `image_scale` and `mode` as the key for `whash`.The default value for `image_scale`
            is equal to max power of 2 for an input image and the `mode` is 'haar', which is Haar wavelets. The other
             possible value for `mode` is db4 - Daubechies wavelets.
         :param is_embed_bool: Set to `True` to encode the images into boolean embeddings using the hashing technique.
         By default set to `False` to encode the images as `np.uint8` embeddings values.
         :param access_paths: The default traversal path on docs, e.g. 'r', 'c'
         :param traversal_paths: please use access_paths
        """
        super().__init__(**kwargs)
        if not hash_type or hash_type not in HASH_TYPE:
            raise ValueError(
                f'Please select one of the available `hash_type`: {HASH_TYPE}'
            )
        self.hash_type = hash_type
        self.hash_size = hash_size
        self._hash_func_args = hash_func_args or {}
        self.is_embed_bool = is_embed_bool
        if traversal_paths is not None:
            self.access_paths = traversal_paths
            warnings.warn("'traversal_paths' will be deprecated in the future, please use 'access_paths'.",
                          DeprecationWarning,
                          stacklevel=2)
        else:
            self.access_paths = access_paths
        self.logger = JinaLogger(
            getattr(self.metas, 'name', self.__class__.__name__)
        ).logger

    @requests
    def encode(self, docs: DocumentArray, parameters: Dict = {}, **kwargs):
        """
        Read the numpy arrays from the Document.tensor, and encode it into embeddings using the
        comparable hashing technique. Features in the image are used to generate a distinct
        (but not unique) fingerprint, and these fingerprints are comparable.

        :param docs: A document array with documents to create embeddings for. Only the
            documents that have the ``blob`` attribute will get embeddings. The ``blob``
            attribute should be the numpy array of the image, and should have dtype
            ``np.uint8``
        :param parameters: A dictionary that contains parameters to control hash encoding.
            The accepted keys are: `hash_type`, `hash_size`, `hash_func_args`, and `access_paths`.
            For example: `parameters={'hash_type': 'phash', 'hash_func_args': {'highfreq_factor': 8}}`
        """
        missing_blob = []
        hash_type = parameters.get('hash_type', self.hash_type)
        hash_size = parameters.get('hash_size', self.hash_size)
        hash_func_args = deepcopy(self._hash_func_args)
        hash_func_args.update(parameters.get('hash_func_args', {}))
        path = parameters.get('access_paths', self.access_paths)

        for doc in docs[path]:
            if doc.tensor is None:
                missing_blob.append(doc.id)
                continue

            image = Image.fromarray(doc.tensor)
            hash_hex = None
            try:
                if hash_type == 'whash':
                    hash_hex = imagehash.whash(
                        image, hash_size=hash_size, **hash_func_args
                    )
                elif hash_type == 'average_hash':
                    hash_hex = imagehash.average_hash(
                        image,
                        hash_size=hash_size,
                    )
                elif hash_type == 'phash':
                    hash_hex = imagehash.phash(
                        image, hash_size=hash_size, **hash_func_args
                    )
                elif hash_type == 'dhash':
                    hash_hex = imagehash.dhash(image, hash_size=hash_size)
                else:
                    raise ValueError(
                        f'Please select one of the available `hash_type`: {HASH_TYPE}'
                    )
            except AssertionError as e:
                self.logger.error(f'Image hashing failed, {e}')

            if hash_hex is not None:
                if self.is_embed_bool:
                    embedding = np.array(hash_hex.hash).reshape(1, hash_size ** 2)
                else:
                    embedding = np.packbits(
                        np.array(hash_hex.hash).reshape(1, hash_size ** 2),
                        axis=1,
                    )
                doc.embedding = np.squeeze(embedding)
            else:
                self.logger.warning(f'Could not set embeddings for {doc.id}')

        if len(missing_blob) > 0:
            self.logger.error(
                f'No blob passed for the following Documents with ids: {missing_blob}'
            )
