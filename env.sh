# source env.sh   (cluster paths; override with VAR=... before sourcing)
export ALPAMAYO_R1="${ALPAMAYO_R1:-/mnt/SHARED-SCRATCH/bthapama/alpamayo-models/Alpamayo-R1-10B}"
export ALPAMAYO_15="${ALPAMAYO_15:-/mnt/SHARED-SCRATCH/bthapama/alpamayo-models/Alpamayo-1.5-10B}"
export GROUNDACT_WORK="${GROUNDACT_WORK:-/mnt/SHARED-SCRATCH/bthapama/groundact-work}"
export HF_HOME="${HF_HOME:-/mnt/SHARED-SCRATCH/bthapama/hf-cache}"
export HF_HUB_OFFLINE=0   # the dataset streams camera video from the Hub
export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd):${PYTHONPATH:-}"
