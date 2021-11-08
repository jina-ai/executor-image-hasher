from jina import Executor, DocumentArray, requests


class ImageHasher(Executor):
    @requests
    def foo(self, docs: DocumentArray, **kwargs):
        pass
