import ast
import io
import tokenize
from collections import defaultdict
from types import ModuleType
from typing import Dict, Set

import black
from datacontract_specification.model import DataContractSpecification, Field, Model

import typespark
from typespark import metadata
from typespark.base import BaseDataFrame


def to_pascal_case(value: str):
    return "".join(value.replace("_", " ").title().split())


def _contract_type_to_pyspark(contract_type: str) -> str:
    """Convert a data contract type string to a PySpark DataType."""
    match contract_type:
        case "string":
            return "String"
        case "int":
            return "Integer"
        case "long":
            return "Long"
        case "short":
            return "Short"
        case "float":
            return "Float"
        case "double":
            return "Double"
        case "bool":
            return "Bool"
        case "timestamp":
            return "TimeStamp"
        case "date":
            return "Date"
        case "bytes":
            return "Binary"
        case "decimal":
            return "Decimal"
        # case "array<string>": ...
        # case "map<string, int>": ...
        # case "struct": ...
        case _:
            raise ValueError(f"Unsupported contract type: {contract_type}")


def _reference(ref: str):

    parts = ref.split(".")

    return ast.Name(to_pascal_case(parts[-2]))


def _field_metadata_ast(field: Field):
    if field.primaryKey and (
        field.type == "decimal" and field.precision and field.scale
    ):
        return ast.Call(
            func=ast.Name(metadata.field.__name__),
            args=[],
            keywords=[
                ast.keyword(arg="primary_key", value=ast.Constant(True)),
                ast.keyword(arg="precision", value=ast.Constant(field.scale)),
                ast.keyword(arg="scale", value=ast.Constant(field.scale)),
            ],
        )
    if field.references and (
        field.type == "decimal" and field.precision and field.scale
    ):
        return ast.Call(
            func=ast.Name(metadata.field.__name__),
            args=[],
            keywords=[
                ast.keyword(arg="foreign_key", value=_reference(field.references)),
                ast.keyword(arg="precision", value=ast.Constant(field.scale)),
                ast.keyword(arg="scale", value=ast.Constant(field.scale)),
            ],
        )
    if field.type == "decimal" and field.precision and field.scale:
        return ast.Call(
            func=ast.Name(metadata.decimal.__name__),
            args=[],
            keywords=[
                ast.keyword(arg="precision", value=ast.Constant(field.scale)),
                ast.keyword(arg="scale", value=ast.Constant(field.scale)),
            ],
        )

    if field.primaryKey:
        return ast.Call(
            func=ast.Name(metadata.primary_key.__name__), args=[], keywords=[]
        )

    if field.references:
        return ast.Call(
            func=ast.Name(metadata.foreign_key.__name__),
            args=[_reference(field.references)],
            keywords=[],
        )

    return None


def generate_field_ast(name: str, field: Field):
    if field.type is not None:
        annotation = ast.Name(id=_contract_type_to_pyspark(field.type), ctx=ast.Load())
    else:
        raise ValueError

    annotation = ast.AnnAssign(
        target=ast.Name(id=name, ctx=ast.Store()),
        annotation=annotation,
        value=_field_metadata_ast(field),
        simple=1,
    )
    if field.description:
        attribute_docstring = ast.Expr(value=ast.Constant(value=field.description))
        return annotation, attribute_docstring

    return (annotation,)


def generate_class_ast(name: str, model: Model):
    if model.description:
        docstring = (ast.Expr(value=ast.Constant(value=model.description)),)
    else:
        docstring = ()

    fields = [generate_field_ast(n, f) for n, f in model.fields.items()]
    body = [element for innerList in fields for element in innerList]

    class_def = ast.ClassDef(
        name=to_pascal_case(name),
        bases=[ast.Name(id=BaseDataFrame.__name__)],
        keywords=[],
        body=[*docstring, *body],
        decorator_list=[],
        type_params=[],
    )

    return class_def


class ImportTracker(ast.NodeVisitor):
    def __init__(self):
        self.used_names = set()
        self.imported_names = set()
        self.function_calls = set()

    def visit_Call(self, node: ast.Call):
        func = node.func
        if isinstance(func, ast.Name):
            self.function_calls.add(func.id)
        elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            self.function_calls.add(f"{func.value.id}.{func.attr}")
        self.generic_visit(node)

    def visit_Name(self, node):
        self.used_names.add(node.id)

    def visit_ImportFrom(self, node):
        for alias in node.names:
            self.imported_names.add(alias.asname or alias.name)

    def visit_Import(self, node):
        for alias in node.names:
            self.imported_names.add(alias.asname or alias.name)


def _module_imports(tree: ast.Module, import_module: ModuleType):
    tracker = ImportTracker()
    tracker.visit(tree)

    # Track symbols to import per module
    pending: Dict[str, Set[str]] = {}

    for name in dir(import_module):
        if name in tracker.used_names and name not in tracker.imported_names:
            pending.setdefault(import_module.__name__, set()).add(name)

    # Insert one import line per module
    for module, symbols in pending.items():
        import_node = ast.ImportFrom(
            module=module, names=[ast.alias(name=s) for s in sorted(symbols)], level=0
        )
        tree.body.insert(0, import_node)


def _get_class_dependencies(class_node: ast.ClassDef) -> set[str]:
    deps = set()

    for stmt in class_node.body:
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.value, ast.Call):
            if (
                isinstance(stmt.value.func, ast.Name)
                and stmt.value.func.id == "foreign_key"
                and stmt.value.args
            ):
                arg = stmt.value.args[0]
                if isinstance(arg, ast.Name):
                    deps.add(arg.id)
    return deps


def _topological_sort_classes(class_defs: list[ast.ClassDef]) -> list[ast.ClassDef]:
    name_to_node = {cls.name: cls for cls in class_defs}
    graph = defaultdict(set)

    for cls in class_defs:
        deps = _get_class_dependencies(cls)
        graph[cls.name].update(dep for dep in deps if dep in name_to_node)

    # Topo sort via DFS
    visited = set()
    ordered = []

    def dfs(n):
        if n in visited:
            return
        visited.add(n)
        for dep in graph[n]:
            dfs(dep)
        ordered.append(n)

    for node in graph:
        dfs(node)

    return [name_to_node[name] for name in ordered]


def _reorder_classes_in_module(module: ast.Module):
    class_defs = [node for node in module.body if isinstance(node, ast.ClassDef)]
    others = [node for node in module.body if not isinstance(node, ast.ClassDef)]

    sorted_classes = _topological_sort_classes(class_defs)
    module.body = others + sorted_classes


def replace_single_with_triple_quotes(code: str) -> str:
    result = []
    paren_level = 0  # tracks nesting inside (), [], {}

    tokens = tokenize.generate_tokens(io.StringIO(code).readline)
    for toknum, tokval, _, __, ___ in tokens:
        if tokval in "([{":
            paren_level += 1
        elif tokval in ")]}":
            paren_level -= 1

        # If it's a single-quoted string outside parentheses
        if (
            toknum == tokenize.STRING
            and paren_level == 0
            and tokval.startswith("'")
            and not tokval.startswith("'''")
        ):
            # Replace '... with '''...'''
            unquoted = tokval[1:-1]  # strip outer quotes
            new_token = f"'''{unquoted}'''"
            result.append((toknum, new_token))
        else:
            result.append((toknum, tokval))

    return tokenize.untokenize(result)


def generate_types(contract: DataContractSpecification):

    module = ast.Module(
        body=[
            generate_class_ast(name, model) for name, model in contract.models.items()
        ],
        type_ignores=[],
    )
    _module_imports(module, metadata)
    _module_imports(module, typespark)
    _reorder_classes_in_module(module)

    return black.format_str(
        replace_single_with_triple_quotes(ast.unparse(module)), mode=black.FileMode()
    )
