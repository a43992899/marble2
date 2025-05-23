# cli.py
from lightning.pytorch.cli import LightningCLI

def main():
    LightningCLI(
        save_config_kwargs={"overwrite": True},
        ) 
    
if __name__ == "__main__":
    main()
