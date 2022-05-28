# DRL_Racers
A2C/PPO DRL Enviroment for Lego Racers (1999)

The Goal: Achieve human performance in a 3D racing environment using only computer vision. The unsupervised environment requires real-time learning and does not allow parallelization. Toolset: Supervised/Unsupervised Reinforcement Learning, CNNs, A2C, PPO, Pytorch.

Project Benchmarks:
    Training Cycles
    Training Time
    Total Reward trend
    Reward Ratio
    Race Time Trend
    Completion Rate
Because we will be generating Millions of cycles, the benchmarks will only be collected every 100 cycles.



        #let's load in our save_states dict.
        print('Opening Evo Dict JSON')
        with open('pretrained_model\evo_dict.json') as json_file:
            loaded_JSON = json.load(json_file)

            #let's convert the place keys from str to int (JSON limitation):
            corrected_dict = {}
            for key, value in loaded_JSON.items():
                value[1] = value[1]+"-X"
                corrected_dict[int(key)] = value

            self.parameters['evo_dict'] = corrected_dict
            print(corrected_dict)
            print('Evo Dict Loaded')



            evo_dict_json = json.dumps(evo_dict)
            with open('pretrained_model\evo_dict.json', 'w') as f:
                f.write(evo_dict_json)
                f.close()