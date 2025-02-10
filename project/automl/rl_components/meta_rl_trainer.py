from ..component import InputSignature, Schema, requires_input_proccess
from .agent_components import AgentSchema
from .optimizer_components import AdamOptimizer
from .exploration_strategy_components import EpsilonGreedyStrategy
from .model_components import ConvModelSchema
from .rl_trainer_component import RLTrainerComponent

import torch
import time

class RLSquaredTrainerComponent(Schema):

    TRAIN_LOG = 'train.txt'
    
    parameters_signature = {"device" : InputSignature(default_value="cpu"),
                       "logger" : InputSignature(),
                       "num_meta_episodes" : InputSignature(),
                       "meta_episode_len" : InputSignature(), 
                       "environment_generator" : InputSignature(),
                       "recurrent_state_size" : InputSignature(),
                       "state_memory_size" : InputSignature(),
                       "agent" : InputSignature(),
                       "limit_steps" : InputSignature(),
                       "optimization_interval" : InputSignature(),
                       "save_interval" : InputSignature(default_value=100),
                       "rl_trainer" : InputSignature(default_value=''),
                       "created_agents_input" : InputSignature(
                            default_value={},
                            description='The input that will be passed to agents created by this pipeline')}
    
    exposed_values = {"total_steps" : 0} #this means we'll have a dic "values" with this starting values


    # INITIALIZATION ----------------------

    def proccess_input(self): #this is the best method to have initialization done right after
        
        super().proccess_input()
        
        self.device = self.input["device"]
        self.lg = self.input["logger"]
        self.lg_profile = self.lg.createProfile(self.name)
        
        
        self.recurrent_state_size = self.input["recurrent_state_size"]
        self.num_meta_episodes = self.input["num_mesa_episodes"]
        self.meta_episode_len = self.input["meta_episode_len"]
        self.step = 0

        self.obs = torch.zeros(
            self.meta_episode_len + 1, self.num_meta_episodes, *observation_space.shape
        )
        self.rewards = torch.zeros(self.meta_episode_len, self.num_meta_episodes, 1)
        
        self.value_preds = torch.zeros(self.meta_episode_len + 1, self.num_meta_episodes, 1)
        
        self.returns = torch.zeros(self.meta_episode_len + 1, self.num_meta_episodes, 1)
        
        self.action_log_probs = torch.zeros(self.meta_episode_len, self.num_meta_episodes, 1)

        # recurrent states
        self.recurrent_states_actor = torch.zeros(
            self.meta_episode_len + 1, self.num_meta_episodes, self.recurrent_state_size
        )
        self.recurrent_states_critic = torch.zeros(
            self.meta_episode_len + 1, self.num_meta_episodes, self.recurrent_state_size
        )

        # masks
        self.done_masks = torch.ones(self.meta_episode_len + 1, self.num_meta_episodes, 1)
         
        
        
    # TRAINING_PROCCESS ----------------------
    
    # TODO: Add seed system
    @requires_input_proccess 
    def train(self):

        # clean
        logging_utils.cleanup_log_dir(self.log_dir)

        rl_squared_envs = make_vec_envs(
            self.config.env_name,
            self.config.env_configs,
            self.config.random_seed,
            self.config.num_processes,
            self.device,
        )

        actor_critic = StatefulActorCritic(
            rl_squared_envs.observation_space,
            rl_squared_envs.action_space,
            recurrent_state_size=256,
        ).to_device(self.device)

        ppo = PPO(
            actor_critic=actor_critic,
            clip_param=self.config.ppo_clip_param,
            opt_epochs=self.config.ppo_opt_epochs,
            num_minibatches=self.config.ppo_num_minibatches,
            value_loss_coef=self.config.ppo_value_loss_coef,
            entropy_coef=self.config.ppo_entropy_coef,
            actor_lr=self.config.actor_lr,
            critic_lr=self.config.critic_lr,
            eps=self.config.optimizer_eps,
            max_grad_norm=self.config.max_grad_norm,
        )

        current_iteration = 0

        # load
        if self._restart_checkpoint:
            checkpoint = torch.load(self._restart_checkpoint)
            current_iteration = checkpoint["iteration"]
            actor_critic.actor.load_state_dict(checkpoint["actor"])
            actor_critic.critic.load_state_dict(checkpoint["critic"])
            ppo.optimizer.load_state_dict(checkpoint["optimizer"])
            pass

        for j in range(current_iteration, self.config.policy_iterations):
            # anneal
            if self.config.use_linear_lr_decay:
                ppo.anneal_learning_rates(j, self.config.policy_iterations)
                pass

            # sample
            meta_episode_batches, meta_train_reward_per_step = sample_meta_episodes(
                actor_critic,
                rl_squared_envs,
                self.config.meta_episode_length,
                self.config.meta_episodes_per_epoch,
                self.config.use_gae,
                self.config.gae_lambda,
                self.config.discount_gamma,
                self.device,
            )

            minibatch_sampler = MetaBatchSampler(meta_episode_batches, self.device)
            ppo_update = ppo.update(minibatch_sampler)

            wandb_logs = {
                "meta_train/mean_policy_loss": ppo_update.policy_loss,
                "meta_train/mean_value_loss": ppo_update.value_loss,
                "meta_train/mean_entropy": ppo_update.entropy,
                "meta_train/approx_kl": ppo_update.approx_kl,
                "meta_train/clip_fraction": ppo_update.clip_fraction,
                "meta_train/explained_variance": ppo_update.explained_variance,
                "meta_train/mean_meta_episode_reward": meta_train_reward_per_step
                * self.config.meta_episode_length,
            }

            # save
            is_last_iteration = j == (self.config.policy_iterations - 1)
            checkpoint_name = str(timestamp()) if self.config.checkpoint_all else "last"

            if j % self.config.checkpoint_interval == 0 or is_last_iteration:
                save_checkpoint(
                    iteration=j,
                    checkpoint_dir=self.config.checkpoint_directory,
                    checkpoint_name=checkpoint_name,
                    actor=actor_critic.actor,
                    critic=actor_critic.critic,
                    optimizer=ppo.optimizer,
                )
                pass

            if enable_wandb:
                wandb.log(wandb_logs)

        # end
        if enable_wandb:
            wandb.finish()
        pass