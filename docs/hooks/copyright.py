from datetime import date


def on_config(config, **kwargs):
    config["copyright"] = f"Copyright &copy; 2024 - {date.today().year} Hugo Vale"
    return config
