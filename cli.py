# cli.py
from lightning.pytorch.cli import LightningCLI

def main():
    LightningCLI(
        save_config_kwargs={"overwrite": True},
        subclass_mode_model=True,
        subclass_mode_data=True,
        ) 
    
if __name__ == "__main__":
    main()
