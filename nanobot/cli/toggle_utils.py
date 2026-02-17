import click


def toggle_feature(feature_name: str, state_obj, attr_name: str):
    current = getattr(state_obj, attr_name)
    click.echo(f"\n{feature_name.upper()} is currently: {'ON' if current else 'OFF'}")
    click.echo("1. ON")
    click.echo("2. OFF")

    choice = click.prompt("Select option", type=int, default=1 if not current else 2)

    if choice == 1:
        setattr(state_obj, attr_name, True)
        click.echo(f"{feature_name.upper()} ENABLED")
        return True
    elif choice == 2:
        setattr(state_obj, attr_name, False)
        click.echo(f"{feature_name.upper()} DISABLED")
        return True
    else:
        click.echo("Invalid choice")
        return False
