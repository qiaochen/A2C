from unityagents import UnityEnvironment
import numpy as np
from agent import ACAgent
from utils import draw

unity_environment_path = "./Reacher_Linux/Reacher.x86_64"
best_model_path = "./best_model.checkpoint"

if __name__ == "__main__":
    # prepare environment
    env = UnityEnvironment(file_name=unity_environment_path)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]
    
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)

    # size of each action
    action_size = brain.vector_action_space_size
    print('Size of each action:', action_size)

    # examine the state space 
    states = env_info.vector_observations
    state_size = states.shape[1]
    print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
    print('The state for the first agent looks like:', states[0])

    num_episodes = 300
    rollout_length = 5
    agent = ACAgent(state_size, 
                    action_size,
                    num_agents,
                    rollout_length=rollout_length,
                    lr=1e-4,
                    lr_decay=.95,
                    gamma=.95,
                    value_loss_weight = 1,
                    gradient_clip = 5,
                    )
    total_rewards = []
    avg_scores = []
    max_avg_score = -1
    max_score = -1
    worsen_tolerance = 10  # for early-stopping training if consistently worsen for # episodes
    rollout = []
    for i_episode in range(1, num_episodes+1):
        env_inst = env.reset(train_mode=True)[brain_name]                       # reset the environment
        states = env_inst.vector_observations                                   # get the current state
        scores = np.zeros(num_agents)                                           # initialize the score
        dones = [False]*num_agents
        steps_taken = 0
        experience = []
        while not np.any(dones):                                                # finish if any agent is done
            steps_taken += 1
            actions, log_probs, state_values = agent.sample_action(states)      # select actions for 20 envs
            env_inst = env.step(actions.detach().cpu().numpy())[brain_name]     # send the actions to the environment
            next_states = env_inst.vector_observations                          # get the next states
            rewards = env_inst.rewards                                          # get the rewards
            dones = env_inst.local_done                                         # see if episode has finished
            not_dones = [1-done for done in dones]        
            experience.append([actions, rewards, log_probs, not_dones, state_values])
            if steps_taken % rollout_length == 0:
                agent.update_model(experience)
                del experience[:]
                
            scores += rewards                                                   # update the scores
            states = next_states                                                # roll over the states to next time step
        episode_score = np.mean(scores)                                         # compute the mean score for 20 agents
        total_rewards.append(episode_score)
        print("Episodic {} Score: {}".format(i_episode, np.mean(scores)))
        if max_score < episode_score:                                           # saving new best model
            max_score = episode_score
            agent.save(best_model_path)
        
        if len(total_rewards) >= 100:                       # record avg score for the latest 100 steps
            latest_avg_score = sum(total_rewards[(len(total_rewards)-100):]) / 100
            print("100 Episodic Everage Score: {}".format(latest_avg_score))
            avg_scores.append(latest_avg_score)
          
            if max_avg_score <= latest_avg_score:           # record better results
                worsen_tolerance = 10                       # re-count tolerance
                max_avg_score = latest_avg_score
                
            else:                                           
                worsen_tolerance -= 1                       # count tolerance
                if max_avg_score > 10:                      # continue from last best-model
                    print("Loaded from last best model.")
                    agent.load(best_model_path)
                if worsen_tolerance <= 0:                   # earliy stop training
                    print("Early Stop Training.")
                    break
                    
    draw(total_rewards,"./training_score_plot.png", "Training Scores (Per Episode)")
    draw(avg_scores,"./training_100avgscore_plot.png", "Training Scores (Average of Latest 100 Episodes)", ylabel="Avg. Score")
    env.close()

                    
