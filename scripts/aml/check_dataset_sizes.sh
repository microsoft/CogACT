#!/bin/bash

# Script to check sizes of Open X-Embodiment datasets
echo "=== Open X-Embodiment Dataset Sizes ==="
echo "Checking sizes for all available datasets..."
echo ""

# Array of all dataset names
datasets=(
    'fractal20220817_data'
    'kuka'
    'bridge'
    'taco_play'
    'jaco_play'
    'berkeley_cable_routing'
    'roboturk'
    'nyu_door_opening_surprising_effectiveness'
    'viola'
    'berkeley_autolab_ur5'
    'toto'
    'language_table'
    'columbia_cairlab_pusht_real'
    'stanford_kuka_multimodal_dataset_converted_externally_to_rlds'
    'nyu_rot_dataset_converted_externally_to_rlds'
    'stanford_hydra_dataset_converted_externally_to_rlds'
    'austin_buds_dataset_converted_externally_to_rlds'
    'nyu_franka_play_dataset_converted_externally_to_rlds'
    'maniskill_dataset_converted_externally_to_rlds'
    'furniture_bench_dataset_converted_externally_to_rlds'
    'cmu_franka_exploration_dataset_converted_externally_to_rlds'
    'ucsd_kitchen_dataset_converted_externally_to_rlds'
    'ucsd_pick_and_place_dataset_converted_externally_to_rlds'
    'austin_sailor_dataset_converted_externally_to_rlds'
    'austin_sirius_dataset_converted_externally_to_rlds'
    'bc_z'
    'usc_cloth_sim_converted_externally_to_rlds'
    'utokyo_pr2_opening_fridge_converted_externally_to_rlds'
    'utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds'
    'utokyo_saytap_converted_externally_to_rlds'
    'utokyo_xarm_pick_and_place_converted_externally_to_rlds'
    'utokyo_xarm_bimanual_converted_externally_to_rlds'
    'robo_net'
    'berkeley_mvp_converted_externally_to_rlds'
    'berkeley_rpt_converted_externally_to_rlds'
    'kaist_nonprehensile_converted_externally_to_rlds'
    'stanford_mask_vit_converted_externally_to_rlds'
    'tokyo_u_lsmo_converted_externally_to_rlds'
    'dlr_sara_pour_converted_externally_to_rlds'
    'dlr_sara_grid_clamp_converted_externally_to_rlds'
    'dlr_edan_shared_control_converted_externally_to_rlds'
    'asu_table_top_converted_externally_to_rlds'
    'stanford_robocook_converted_externally_to_rlds'
    'eth_agent_affordances'
    'imperialcollege_sawyer_wrist_cam'
    'iamlab_cmu_pickup_insert_converted_externally_to_rlds'
    'uiuc_d3field'
    'utaustin_mutex'
    'berkeley_fanuc_manipulation'
    'cmu_food_manipulation'
    'cmu_play_fusion'
    'cmu_stretch'
    'berkeley_gnm_recon'
    'berkeley_gnm_cory_hall'
    'berkeley_gnm_sac_son'
)

total_size_bytes=0
found_count=0
not_found_count=0

echo "Dataset Name | Size | Status"
echo "-------------|------|-------"

for dataset in "${datasets[@]}"; do
    # Try to get the size of the dataset
    size_output=$(gsutil du -s gs://gresearch/robotics/$dataset/0.1.0 2>/dev/null)
    
    if [ $? -eq 0 ] && [ -n "$size_output" ]; then
        # Extract size in bytes
        size_bytes=$(echo "$size_output" | awk '{print $1}')
        
        # Convert to human readable
        if [ "$size_bytes" -gt 1073741824 ]; then
            size_gb=$(echo "scale=2; $size_bytes / 1073741824" | bc)
            size_human="${size_gb} GB"
        elif [ "$size_bytes" -gt 1048576 ]; then
            size_mb=$(echo "scale=2; $size_bytes / 1048576" | bc)
            size_human="${size_mb} MB"
        elif [ "$size_bytes" -gt 1024 ]; then
            size_kb=$(echo "scale=2; $size_bytes / 1024" | bc)
            size_human="${size_kb} KB"
        else
            size_human="${size_bytes} B"
        fi
        
        printf "%-40s | %10s | %s\n" "$dataset" "$size_human" "✓ Found"
        total_size_bytes=$(echo "$total_size_bytes + $size_bytes" | bc)
        found_count=$((found_count + 1))
    else
        printf "%-40s | %10s | %s\n" "$dataset" "N/A" "✗ Not found/Access denied"
        not_found_count=$((not_found_count + 1))
    fi
done

echo ""
echo "=== Summary ==="
echo "Found datasets: $found_count"
echo "Not found/No access: $not_found_count"

if [ "$total_size_bytes" -gt 0 ]; then
    total_gb=$(echo "scale=2; $total_size_bytes / 1073741824" | bc)
    echo "Total size of accessible datasets: ${total_gb} GB"
fi

echo ""
echo "=== Recommendations ==="
echo "For large datasets (>10GB), consider downloading in stages or specific subsets."
echo "Use 'gsutil ls -l gs://gresearch/robotics/DATASET_NAME/0.1.0/' to see individual files."
