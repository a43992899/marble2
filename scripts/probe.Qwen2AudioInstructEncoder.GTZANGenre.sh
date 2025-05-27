rm -rf output/probe.GTZANGenre.Qwen2AudioInstructEncoder
python cli.py fit -c configs/probe.Qwen2AudioInstructEncoder.GTZANGenre.yaml
python cli.py test -c configs/probe.Qwen2AudioInstructEncoder.GTZANGenre.yaml