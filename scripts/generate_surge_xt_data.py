import click


@click.command()
@click.option("--surge-path", default="vsts/Surge XT.vst3")
def main(surge_path: str):
    pass


if __name__ == "__main__":
    main()
