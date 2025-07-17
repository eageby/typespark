try:
    import simple_parsing

except ImportError as e:
    raise ImportError(
        "The typespark[schema] package is required for this feature. "
        "Please install it with `pip install typespark[schema]`."
    ) from e


from .schema import generate_schema
