rm -rf output/probe.GS.MERT-v1-95M
python cli.py fit -c configs/probe.MERT-v1-95M.GS.yaml
python cli.py test -c configs/probe.MERT-v1-95M.GS.yaml