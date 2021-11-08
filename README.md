# ImageHasher


## Usage

#### via Docker image (recommended)

```python
from jina import Flow
	
f = Flow().add(uses='jinahub+docker://ImageHasher')
```

#### via source code

```python
from jina import Flow
	
f = Flow().add(uses='jinahub://ImageHasher')
```

- To override `__init__` args & kwargs, use `.add(..., uses_with: {'key': 'value'})`
- To override class metas, use `.add(..., uses_metas: {'key': 'value})`
