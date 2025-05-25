rm -rf output/probe.GTZANGenre.MERT-v1-95M
python cli.py fit -c configs/probe.MERT-v1-95M.GTZANGenre.yaml
python cli.py test -c configs/probe.MERT-v1-95M.GTZANGenre.yaml