rm -rf output/finetune.GTZANGenre.MERT-v1-95M
python cli.py fit -c configs/finetune.MERT-v1-95M.GTZANGenre.yaml
python cli.py test -c configs/finetune.MERT-v1-95M.GTZANGenre.yaml