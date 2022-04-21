python barl_run.py -m env_config_file=config/envs/pendulum.yaml agent=bptt name=hucrl_pendulum_thompson exploration=thompson seed="range(5)" hydra/launcher=joblib
python barl_run.py -m env_config_file=config/envs/pendulum.yaml agent=bptt name=hucrl_pendulum_greedy exploration=greedy seed="range(5)" hydra/launcher=joblib
python barl_run.py -m env_config_file=config/envs/pendulum.yaml agent=bptt name=hucrl_pendulum exploration=optimistic seed="range(5)" hydra/launcher=joblib
