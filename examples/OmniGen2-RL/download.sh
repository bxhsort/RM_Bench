export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --repo-type dataset --resume-download EditScore/EditScore-RL-Data --local-dir /home/nas/data/Data/EditScore-RL-Data
huggingface-cli download OmniGen2/OmniGen2 --local-dir /home/nas/data/RM_Bench/OmniGen2
huggingface-cli download --resume-download EditScore/EditScore-7B --local-dir /home/nas/data/RM_Bench/ckpts/EditScore-7B
huggingface-cli download --resume-download Qwen/Qwen2.5-VL-7B-Instruct --local-dir /home/nas/data/RM_Bench/ckpts/Qwen2.5-VL-7B-Instruct
huggingface-cli download --resume-download Qwen/Qwen3-VL-8B-Instruct --local-dir /home/nas/data/RM_Bench/ckpts/Qwen3-VL-8B-Instruct
huggingface-cli download --resume-download EditScore/EditScore-Qwen3-VL-8B-Instruct --local-dir /home/nas/data/RM_Bench/ckpts/EditScore-Qwen3-VL-8B-Instruct