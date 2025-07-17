try:
    import datacontract_specification
except ImportError as e:
    raise ImportError(
        "The typespark[datacontract] package is required for this feature. "
        "Please install it with `pip install typespark[datacontract]`."
    ) from e

from .serialize import serialize_product
from .generate import generate_types
