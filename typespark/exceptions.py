from typespark.columns.columns import TypedColumn


class MissingColumnError(ValueError):
    def __init__(
        self,
        *,
        model: type,
        field_name: str,
        expected_column: str,
        available_columns: list[str] | None = None,
    ):
        self.model = model
        self.field_name = field_name
        self.expected_column = expected_column
        self.available_columns = available_columns

        msg = (
            f"Missing required column '{expected_column}' for field '{field_name}' "
            f"in {model.__name__}. "
        )
        if available_columns is not None:
            msg += f"Available columns: {available_columns}. "
        msg += f"To fix: provide the column, adjust aliases, or set a default for '{field_name}'."
        super().__init__(msg)


class InvalidDefaultColumnError(TypeError):
    def __init__(
        self,
        *,
        model: type,
        field_name: str,
        field_alias: str,
        default_value: object,
    ):
        self.model = model
        self.field_name = field_name
        self.field_alias = field_alias
        self.default_value = default_value

        msg = (
            f"Default for {model.__name__}.{field_name} must be a "
            f"{TypedColumn.__name__}. "
            f"Got {type(default_value).__name__}: {default_value!r}. "
            f"Field alias: {field_alias!r}."
        )
        super().__init__(msg)
