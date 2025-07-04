source /Users/arkady/Desktop/dev/diamond/venv/bin/activate


python collect_static_dataset.py \
    /Users/arkady/Desktop/dev/diamond/outputs/NoWorldEnv/checkpoints/agent_versions/agent_epoch_00050.pt \
    /Users/arkady/Desktop/dev/diamond/dataset/aNoWorldEnv050/001eps_100Ksteps \
    --steps-to-collect 100000 \
    --epsilon 0.01