#!/bin/bash


# -- helper
echo -e "## Train Dust3r \n"

# Function to display the command help
show_help() {
    echo    "## Usage: $0 [options]"
    echo -e "          $0 <output_dir> \n"
    echo
    echo "Options:"
    echo "  --help     Show this help message and exit"
    echo
}

# Check for the --help flag
if [[ -z "$1" || "$1" == "-h" || "$1" == "--help" || "$1" == "--helper" ]]; then
    show_help
    exit 0
fi

OUTPUT_DIR="$1"


# -- set variables
ROOT="/srv/essa-lab/flash3/vcartillier3/egoexo-view-synthesis/dependencies/dust3r"

SCRIPT_DIR="$ROOT/batch_scripts"

RUNNER_WRAPPER_SCRIPT="$SCRIPT_DIR/template_wrapper.sh"

UNIQUE_ID=$((1 + RANDOM % 100000))

# -- grab un-usable nodes
mapfile -t all_nodes < <( squeue -u abeedu3 -o "%N" )

all_nodes=("${all_nodes[@]:1}")

unique_nodes=($(printf '%s\n' "${all_nodes[@]}" | sort -u))

exclude_nodes=""
for x in "${unique_nodes[@]}"; do
    exclude_nodes+="$x,"
done

exclude_nodes="${exclude_nodes:0:-1}"
exclude_nodes="voltron,xaea-12"
echo -e "## excluding these nodes: $exclude_nodes \n"

echo "output dir: $OUTPUT_DIR"

# create copy of runner_wrapper.sh template
tmp_wrapper_name="wrapper_tmp_${UNIQUE_ID}.sh"
dst_wrapper="$SCRIPT_DIR/tmp/$tmp_wrapper_name"

cp "$RUNNER_WRAPPER_SCRIPT" "$dst_wrapper"

chmod +x "$dst_wrapper"

sbatch --exclude="$exclude_nodes" slurm_script.sh "$dst_wrapper" "$OUTPUT_DIR"



