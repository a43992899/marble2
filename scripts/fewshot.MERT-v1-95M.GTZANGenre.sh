rm -rf output/fewshot.GTZANGenre.MERT-v1-95M
python cli.py fit -c configs/fewshot.MERT-v1-95M.GTZANGenre.yaml
python cli.py test -c configs/fewshot.MERT-v1-95M.GTZANGenre.yaml