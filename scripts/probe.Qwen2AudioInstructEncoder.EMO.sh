rm -rf output/probe.EMO.MERT-v1-95M
python cli.py fit -c configs/probe.Qwen2AudioInstructEncoder.EMO.yaml
python cli.py test -c configs/probe.Qwen2AudioInstructEncoder.EMO.yaml