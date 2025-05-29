rm -rf output/probe.GTZANGenre.Qwen2AudioInstructEncoder
CUDA_VISIBLE_DEVICES=2 python cli.py fit -c configs/probe.Qwen2AudioInstructEncoder.GS.yaml
CUDA_VISIBLE_DEVICES=2 python cli.py test -c configs/probe.Qwen2AudioInstructEncoder.GS.yaml