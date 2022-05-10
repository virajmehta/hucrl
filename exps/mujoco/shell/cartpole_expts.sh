python barl_run.py env_config_file=config/envs/barl-cartpole.yaml agent=bptt name=hucrl_cartpole_thompson exploration=thompson seed="range(5)" hydra/launcher=joblib
python barl_run.py env_config_file=config/envs/barl-cartpole.yaml agent=bptt name=hucrl_cartpole_greedy exploration=greedy seed="range(5)" hydra/launcher=joblib
python barl_run.py env_config_file=config/envs/barl-cartpole.yaml agent=bptt name=hucrl_cartpole exploration=optimistic seed="range(5)" hydra/launcher=joblib
