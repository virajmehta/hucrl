"""Script that demonstrates how to use BPTT using hallucination."""

import argparse
import importlib
import hydra

from rllib.environment import GymEnvironment
from rllib.model import TransformedModel
from rllib.util import set_random_seed
from rllib.util.training.agent_training import evaluate_agent, train_agent

from exps.util import parse_config_file
from hucrl.environment.hallucination_wrapper import HallucinationWrapper
from hucrl.model.hallucinated_model import HallucinatedModel
from barl import envs
from hucrl.reward.mujoco_rewards import barl_reward_models


@hydra.main(config_path="cfg", config_name="config")
def main(args):
    """Run experiment."""
    set_random_seed(args.seed)
    env_config = parse_config_file(args.env_config_file)

    environment = GymEnvironment(
        env_config["name"], seed=args.seed
    )
    reward_model = barl_reward_models[env_config['name']]()
    if args.exploration == "optimistic":
        dynamical_model = HallucinatedModel.default(environment, beta=args.beta)
        environment.add_wrapper(HallucinationWrapper)
    else:
        dynamical_model = TransformedModel.default(environment)
    kwargs = parse_config_file(args.agent_config_file)

    agent = getattr(
        importlib.import_module("rllib.agent"), f"{args.agent}Agent"
    ).default(
        environment=environment,
        dynamical_model=dynamical_model,
        reward_model=reward_model,
        thompson_sampling=args.exploration == "thompson",
        **kwargs,
    )
    train_agent(
        agent=agent,
        environment=environment,
        max_steps=env_config["max_steps"],
        num_episodes=args.train_episodes,
        render=args.render,
        eval_frequency=1,
        print_frequency=1,
    )

    evaluate_agent(
        agent=agent,
        environment=environment,
        max_steps=env_config["max_steps"],
        num_episodes=args.test_episodes,
    )


if __name__ == "__main__":
    main()
