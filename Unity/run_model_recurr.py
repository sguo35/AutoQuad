from mlagents.envs import UnityEnvironment, BrainInfo
import numpy as np
import time
import sys

from logger import Logger

from yaml_loader import read_params


from strategy import RandomArcStrategy, EGreedyArcStrategy

#utility

# true_print = print
# def print(*args):
#     true_print(*args)
#     sys.stdout.flush()

class RunAgent:

    # args is either params dict or a yaml filename
    def __init__(self, agent, args, demonstrations=False):

        if not isinstance(args, dict):
            #read as yaml file
            args = read_params(args)

        self._agent = agent
        if demonstrations:
            self.env_name = args['system']['demonstration_drone_sim']
        else:
            self.env_name = args['system']['drone_sim']
        self._args = args
        self.create_env(self.env_name)
        self.parseArgs(self._args)

        self.lg = Logger(self.log_prefix, display_name="run", console_lvl=self.console_lvl)

    def parseArgs(self, args):
        #training
        self.max_episode_length = args['train']['max_episode_length']
        self.num_episodes = args['train']['num_episodes']
        self.batch_size = args['train']['batch_size']
        self.train_period = args['train']['train_period']
        self.train_on_demonstrations = args['train']['train_on_demonstrations']
        self.demonstration_eval_episodes = args['train']['demonstration_eval_episodes']
        self.demonstration_epochs = args['train']['demonstration_epochs']
        self.train_after_episode = args['train']['train_after_episode']
        self.reinforce_good_episodes = args['train']['reinforce_good_episodes']
        self.good_ep_thresh = args['train']['good_ep_thresh']
        self.model_file = args['train']['model_file']
        #inference
        self.only_inference = args['inference']['only_inference']
        self.num_inference_episodes = args['inference']['num_inference_episodes']
        #demonstrations

        #logging
        self.log_prefix = args['logging']['log_prefix']
        self.console_lvl = args['logging']['console_lvl']

    def create_env(self, file_name):
        self._env = UnityEnvironment(file_name=None, worker_id=0)


    #must pass in exploration_strategy object for training
    def run(self, load=False, exploration_strategy=None):
        if self.only_inference:
            self.lg.print("-- Running INFERENCE on %d episodes of length %d -- \n" % (self.num_inference_episodes, self.max_episode_length), lvl="info")
            self.run_inference()
        else:
            self.lg.print("-- Running TRAINING on %d episodes of length %d -- \n" % (self.num_episodes, self.max_episode_length), lvl="info")
            self.run_training(load=load,
                              exploration_strategy=exploration_strategy)

    #gives them pretty high reward
    def train_demonstrations(self):
        pass
 
    #must pass in exploration_strategy object
    def run_training(self, sim_train_mode=True, load=False,
                     exploration_strategy=None): #batch_size=32, num_episodes=1, max_episode_length=1000, train_period=3, train_after_episode=False, train_mode=False):

        if load:
            try:
                self._agent.load(self.model_file)
            except Exception as e:
                self.lg.print("Could not load from file:", str(e), "error")

        if self.train_on_demonstrations:
            success = self.train_demonstrations()
            if success:
                self.dem_ep_count = 0 #basically runs trajectory mostly greedily first

        for e in range(self.num_episodes):
            walltime = time.time()

            #reset
            brainInf = self._env.reset(train_mode=sim_train_mode)['DroneBrain']

            p_observation = self._agent.preprocess_observation(brainInf.visual_observations[0])

            vec_obs = np.zeros(shape=(self.max_episode_length, 5))
            vec_obs[0] = brainInf.vector_observations[0]

            rewards = []
            done = False

            self.lg.print("-- Episode %d --" % e)
            sys.stdout.flush()

            greedy = e < self.demonstration_eval_episodes

            episode_samples = []

            if exploration_strategy is not None:
                trajectory = exploration_strategy.generate_trajectory(args={})
            else:
                trajectory = None

            for t in range(self.max_episode_length):
                timestep_start = time.time()
                #generalized act function takes in state and observations (images)
                #import pdb
                #pdb.set_trace()


                if trajectory is None:
                    action = self._agent.act(vec_obs, p_observation, greedy=greedy)
                else:
                    #use exploration strategy
                    action = next(trajectory)
                nextBrainInf = self._env.step(action)['DroneBrain']

                done = brainInf.local_done[0]
                #self.lg.print(brainInf.local_done)
                reward = self._agent.compute_reward(brainInf, nextBrainInf, action)
                rewards.append(reward)

                stored_example = (np.copy(vec_obs), np.copy(p_observation))
                p_observation = self._agent.preprocess_observation(nextBrainInf.visual_observations[0])
                vec_obs[t] = nextBrainInf.vector_observations[0]

                if t % 4 == 0:
                    #stores processed things
                    sample = (  stored_example, 
                                action,
                                reward,
                                (np.copy(vec_obs), np.copy(p_observation)),
                                done    )

                    episode_samples.append(sample)
                    self._agent.store_sample(sample)

                #train every experience here
                if not self.train_after_episode and len(self._agent.replay_buffer) > self.batch_size:
                    if t % self.train_period == 0:
                        self._agent.train(self.batch_size)

                if t % 100 == 0:
                    #self.lg.print("step", t)
                    sys.stdout.flush()

                if done:
                    break

                brainInf = nextBrainInf
                #print("Step took", (time.time() - timestep_start) / 1000, "milliseconds")

            #LOGGING
            episode_str = ( "Episode {}/{} completed,"
                           " \n\t total steps: {}," 
                           " \n\t total reward: {},"
                           " \n\t mean reward: {},"
                           " \n\t max reward: {},"
                           " \n\t min reward: {},"
                           " \n\t greedy: {},"
                           " \n\t epsilon: {},"
                           " \n\t sim time: {}" )
            self.lg.print(episode_str.format(e, self.num_episodes, t, np.sum(rewards),
                  np.mean(rewards), np.max(rewards), np.min(rewards), greedy,
                             self._agent.epsilon, time.time() - walltime), lvl="info")
            # train after episode
            if self.train_after_episode and len(self._agent.replay_buffer) > self.batch_size:
                walltime = time.time()
                for i in range(t // self.train_period):
                    self._agent.train(self.batch_size)
                self.lg.print("training complete in: {}".format(time.time() - walltime))
                sys.stdout.flush()

            #good episode reinforcement
            if np.sum(rewards) >= self.good_ep_thresh:
                print("-- SUCCESSFUL EPISODE. Training More -- ")
                new_batch_size = min(len(self._agent.replay_buffer), self.batch_size * self.reinforce_good_episodes)
                self._agent.train(new_batch_size)

            # save after an episode
            if e % 10 == 0:
                self._agent.save(self.model_file)

            # update epsilon after each episode
            self._agent.epsilon_update()

        self.lg.print("|------------| TRAINING COMPLETE |------------|", lvl="info")

    def run_inference(self, train_mode=False):
        self._agent.load(self.model_file)

        #reset
        for e in range(self.num_inference_episodes):
            walltime = time.time()

            #reset
            brainInf = self._env.reset(train_mode=train_mode)['DroneBrain']
            # import ipdb; ipdb.set_trace()

            p_observation = self._agent.preprocess_observation(brainInf.visual_observations[0])

            rewards = []
            done = False

            self.lg.print("-- Episode %d --" % e)

            for t in range(self.max_episode_length):
                #generalized act function takes in state and observations (images)

                action = self._agent.act(brainInf.vector_observations,
                                         p_observation, greedy=True)

                nextBrainInf = self._env.step(action)['DroneBrain']

                done = brainInf.local_done[0]
                # self.lg.print(brainInf.local_done)
                reward = self._agent.compute_reward(brainInf, nextBrainInf, action)
                rewards.append(reward)

                if done:
                   break

                brainInf = nextBrainInf

            #LOGGING
            episode_str = ( "Episode {}/{} completed,"
                           " \n\t total steps: {}," 
                           " \n\t total reward: {},"
                           " \n\t mean reward: {},"
                           " \n\t max reward: {},"
                           " \n\t min reward: {},"
                           " \n\t epsilon: {},"
                           " \n\t sim time: {}" )
            # self.lg.print("Episode {}/{} completed, \n\t total steps: {},\n\t total reward: {},\n\t mean reward: {},\n\t sim time: {}".format(e, self.num_episodes, t, np.sum(rewards), np.mean(rewards), time.time() - walltime))
            self.lg.print(episode_str.format(e, self.num_episodes, t, np.sum(rewards),
                  np.mean(rewards), np.max(rewards), np.min(rewards),
                    self._agent.epsilon, time.time() - walltime), lvl="info")

        # self.lg.print("---------------------------")
        # self.lg.print("||  Inference Completed  ||")
        self.lg.print("|------------| INFERENCE COMPLETE |------------|", lvl="info")


    def run_demonstrations(self, train_mode=False, load=True):
        pass




