import sys
sys.path.insert(0,"/root/qmix")
from runner import Runner
from env.starcraftenv import StarCraft2Env
from arguments import get_common_args, get_mixer_args 

if __name__ == '__main__':
    for i in range(8):
        args = get_common_args()
        args = get_mixer_args(args)
        env = StarCraft2Env(map_name=args.map_name,
                            step_mul=args.step_mul,
                            difficulty=args.difficulty,
                            game_version=args.game_version,
                            replay_dir=args.replay_dir)
        env_info = env.get_env_info()
        args.n_actions = env_info["n_actions"]
        args.n_agents = env_info["n_agents"]
        args.state_shape = env_info["state_shape"]
        print("No of Action  = ",env_info["n_actions"])
        print("No of Agents  = ",env_info["n_agents"])
        print("Obs shape = ",env_info["obs_shape"])
        print("Episode_limit = ",env_info["episode_limit"])
        args.obs_shape = env_info["obs_shape"]
        args.episode_limit = env_info["episode_limit"]
        args.alg = "qmix"    
        args.attack_name = "strategic"    #"random"   "random_time"  "strategic"
        args.strategic_threshold = 0.55   #set threshold for strategic time attack
        args.victim_agent= 2              # victim agent to attack
        args.attack_rate = 0.25           #Theshold value for the frequency of attack
        args.adversary = True;            #False = Optimal Qmix; True = If you want to enforce any of the attack ("random"   "random_time"  "strategic") 
        runner = Runner(env, args)
        if not args.evaluate:
            runner.run(i)
        else:
            win_rate, _ = runner.evaluate()
            print('The win rate of {} is  {}'.format(args.alg, win_rate))
            break
        env.close()
