from typespark.columns.columns import TypedColumn


class UnnamedColumnError(ValueError):
    def __init__(self, *, column_type: type, operation: str):
        self.column_type = column_type
        self.operation = operation

        msg = (
            f"Cannot call '{operation}' on an unnamed {column_type.__name__} column. "
            f"Use a column from a typespark DataFrame, or call ._set_name() to assign a name."
        )
        super().__init__(msg)


class MissingColumnError(ValueError):
    def __init__(
        self,
        *,
        model: type,
        field_name: str,
        expected_column: str,
        available_columns: list[str] | None = None,
        parent_field: str | None = None,
    ):
        self.model = model
        self.field_name = field_name
        self.expected_column = expected_column
        self.available_columns = available_columns
        self.parent_field = parent_field

        msg = (
            f"Missing required column '{expected_column}' for field '{field_name}' "
            f"in {model.__name__}"
        )
        if parent_field is not None:
            msg += f" (nested under struct column '{parent_field}')"
        msg += ". "
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
