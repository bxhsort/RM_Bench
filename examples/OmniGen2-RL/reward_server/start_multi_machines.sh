# !/bin/bash
# SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)

# root_dir=$SHELL_FOLDER
# model_name=editscore_7B
# config_path=${root_dir}/server_configs/editscore_7B.yml

# while [[ $# -gt 0 ]]; do
#     case "$1" in
#         --config_path=*)
#             config_path="${1#*=}"
#             shift
#             ;;
#         --model_name=*)
#             model_name="${1#*=}"
#             shift
#             ;;
#         *)
#             echo "Unknown parameter: $1"
#             shift
#             ;;
#     esac
# done

# hosts_string=$(python ${root_dir}/scripts/misc/load_host.py --config_path=${config_path})
# hosts=(${hosts_string})

# SCRIPT="bash ${root_dir}/step1.sh --config_path=${config_path} --model_name=${model_name}"

# for i in "${!hosts[@]}"; do
#     echo "$i ${hosts[$i]}"
#     ssh -o StrictHostKeyChecking=no "${hosts[$i]}" "tmux new-session -d -s reward_server '$SCRIPT --machine_id=$i' -p 4795" &
# done

# SCRIPT="bash ${root_dir}/step2.sh --config_path=${config_path} --model_name=${model_name}"

# for i in "${!hosts[@]}"; do
#     echo "$i ${hosts[$i]}"
#     ssh -o StrictHostKeyChecking=no "${hosts[i]}" "tmux new-session -d -s reward_proxy '$SCRIPT --machine_id=$i' -p 4795"  &
# done

# echo "step2.sh started successfully"

# wait
# echo "All tasks started successfully"

# !/bin/bash
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)

root_dir=$SHELL_FOLDER
model_name=editscore_7B
config_path=${root_dir}/server_configs/editscore_7B.yml

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config_path=*)
            config_path="${1#*=}"
            shift
            ;;
        --model_name=*)
            model_name="${1#*=}"
            shift
            ;;
        *)
            echo "Unknown parameter: $1"
            shift
            ;;
    esac
done

hosts_string=$(python ${root_dir}/scripts/misc/load_host.py --config_path=${config_path})
hosts=(${hosts_string})

SCRIPT="bash ${root_dir}/step1.sh --config_path=${config_path} --model_name=${model_name}"

for i in "${!hosts[@]}"; do
    echo "$i ${hosts[$i]}"
    host_entry="${hosts[$i]}"
    port="4795"
    host_no_port="$host_entry"
    # support formats: host:port or user@host:port
    if [[ "$host_entry" == *:* ]]; then
        port="${host_entry##*:}"
        host_no_port="${host_entry%:*}"
    fi
    if [[ -n "$port" ]]; then
        ssh -o StrictHostKeyChecking=no -p "${port}" "${host_no_port}" "tmux new-session -d -s reward_server '$SCRIPT --machine_id=$i'" &
    else
        ssh -o StrictHostKeyChecking=no "${host_no_port}" "tmux new-session -d -s reward_server '$SCRIPT --machine_id=$i'" &
    fi
done

SCRIPT="bash ${root_dir}/step2.sh --config_path=${config_path} --model_name=${model_name}"

for i in "${!hosts[@]}"; do
    echo "$i ${hosts[$i]}"
    host_entry="${hosts[$i]}"
    port="4795"
    host_no_port="$host_entry"
    # support formats: host:port or user@host:port
    if [[ "$host_entry" == *:* ]]; then
        port="${host_entry##*:}"
        host_no_port="${host_entry%:*}"
    fi
    if [[ -n "$port" ]]; then
        ssh -o StrictHostKeyChecking=no -p "${port}" "${host_no_port}" "tmux new-session -d -s reward_proxy '$SCRIPT --machine_id=$i'" &
    else
        ssh -o StrictHostKeyChecking=no "${host_no_port}" "tmux new-session -d -s reward_proxy '$SCRIPT --machine_id=$i'" &
    fi
done

echo "step2.sh started successfully"

wait
echo "All tasks started successfully"