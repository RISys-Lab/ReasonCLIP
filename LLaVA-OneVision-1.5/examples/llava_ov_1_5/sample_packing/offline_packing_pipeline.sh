#!/bin/bash
CONTAINER_NAME="your_container_name"
docker start "$CONTAINER_NAME"
docker exec -it "$CONTAINER_NAME" bash -c '
    set -e
    set -u
    run_python_script() {
        local script_name=$1
        echo ">>>>>>>>>>>start execution $script_name >>>>>>>>>>>>>>"
        python "$script_name"
        echo ">>>>>>>>>>>$script_name execution completed>>>>>>>>>>>>>>"
    }
    cd /your_LLaVA-OneVision-1.5_path/llava_ov_1_5/sample_packing
    
    run_python_script "huggingface_data_parse.py"
    run_python_script "1_s1_get_tokenlens_v3-sft.py"
    run_python_script "2_do_hashbacket.py"
    run_python_script "3_s2_prepare_rawsamples-vqa.py"
    run_python_script "4_convert_packedsample_to_wds.py"

    echo "─────────────────All processing workflows have been successfully completed.───────────────────"
'