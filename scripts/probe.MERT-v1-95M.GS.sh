rm -rf output/probe.GS.MERT-v1-95M
CUDA_VISIBLE_DEVICES=1 python cli.py fit -c configs/probe.MERT-v1-95M.GS.yaml
CUDA_VISIBLE_DEVICES=1 python cli.py test -c configs/probe.MERT-v1-95M.GS.yaml