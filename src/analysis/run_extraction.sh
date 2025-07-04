source /Users/arkady/Desktop/dev/diamond/venv/bin/activate
nohup python extract_trajectories.py \
    /Users/arkady/Desktop/dev/diamond/outputs/NoWorldEnv/checkpoints/agent_versions/agent_epoch_00000.pt \
    /Users/arkady/Desktop/dev/diamond/outputs/PaperLike/checkpoints/agent_versions/agent_epoch_00950.pt \
    /Users/arkady/Desktop/dev/diamond/analysis_data/extracted_trajectories/aNoWorldEnv000_dPaperLike950 \
    --autoregresive-length 100 \ 
    --episodes-to-collect 100