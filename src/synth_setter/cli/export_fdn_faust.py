"""CLI for exporting fixed-value Faust source from a pyFDN build."""

from pathlib import Path

import click

from synth_setter.data.fdn_faust import export_fdn_faust


@click.command(help="Export a pyFDN v2 build as compiled fixed-value Faust source.")
@click.argument("input_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("output_path", type=click.Path(dir_okay=False, path_type=Path))
def main(input_path: Path, output_path: Path) -> None:
    """Export INPUT_PATH as a compiled fixed-value Faust OUTPUT_PATH.

    :param input_path: Existing pyFDN v2 build JSON.
    :param output_path: New ``.dsp`` destination.
    :raises click.ClickException: The build cannot be exported safely.
    """
    try:
        export_fdn_faust(input_path, output_path)
    except FileExistsError as err:
        raise click.ClickException(f"refusing to overwrite existing {output_path}") from err
    except (OSError, RuntimeError, TypeError, ValueError) as err:
        raise click.ClickException(str(err)) from err
    click.echo(f"Exported fixed Faust DSP: {input_path} -> {output_path}")


if __name__ == "__main__":
    main()
