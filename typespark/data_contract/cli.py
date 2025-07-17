from pathlib import Path

import click
from datacontract_specification.model import DataContractSpecification

from typespark.data_contract.generate import generate_types


@click.group("typespark")
def main():
    pass


@main.command()
@click.argument("contract", type=Path)
@click.option(
    "--output",
    type=Path,
    default=None,
    help="Output path, will generate from data contract title if not provided",
)
def generate(contract: Path, output: Path | None):
    """Generate Typespark types from data contract specification YAML file"""

    contract_model = DataContractSpecification.from_file(str(contract))

    if (
        output is None
        and contract_model.info is not None
        and contract_model.info.title is not None
    ):
        output = Path(contract_model.info.title.lower() + "_models.py")
    if output is None:
        raise ValueError(
            "Can't resolve output file name from data contract, please provided using --output"
        )

    with output.open("w", encoding="utf-8") as output_file:
        output_file.write(generate_types(contract_model))
