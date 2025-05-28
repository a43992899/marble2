rm -rf output/probe.EMO.MERT-v1-95M
python cli.py fit -c configs/probe.MERT-v1-95M.EMO.yaml
python cli.py test -c configs/probe.MERT-v1-95M.EMO.yaml