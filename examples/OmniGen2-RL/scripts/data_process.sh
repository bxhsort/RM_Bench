# Then
python scripts/data/process_jsonl.py --input /home/nas/data/Data/EditScore-RL-Data/rl.jsonl --output /home/nas/data/Data/EditScore-RL-Data/rl_abs.jsonl --base-path /home/nas/data/Data/EditScore-RL-Data

# Due to the limitation of base model (OmniGen2), we discard text change and portrait beautification, as these tasks harm RL training.
python scripts/data/extract_9_tasks.py --input_path /home/nas/data/Data/EditScore-RL-Data/rl_abs.jsonl --output_path /home/nas/data/Data/EditScore-RL-Data/rl_abs_9tasks.jsonl