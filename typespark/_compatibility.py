import importlib


def check_pyspark_version(min_version="3.5.2", max_version="4.0.0"):
    try:
        pyspark = importlib.import_module("pyspark")

    except ImportError:
        raise ImportError(
            "PySpark is required but not installed. "
            "Install via `pip install pyspark` or use the `[typespark]` extra."
        )

    from packaging.version import parse

    version = parse(pyspark.__version__)
    if version < parse(min_version) or version >= parse(max_version):
        raise RuntimeError(
            f"Unsupported PySpark version {version}. "
            f"Supported range: >= {min_version}, < {max_version}."
        )
