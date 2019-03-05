
# gym-act

The [Anti Collision Tests (ACT) environment](https://github.com/PhilippeW83440/CS234_Project/tree/master/act-env) is a driver agent
task featuring continuous state and action spaces.  
The driver agent is pursuing multiple objectives:
* Efficiency: time to goal shall be minimized
* Comfort: the number of hard braking decisions shall be minimized  
* Safety: collisions shall be avoided) shall be optimized 
 The Driver Models of other cars in the scene are the main source of uncertainty to deal with.


# Installation

```bash
cd gym-act
pip install -e .
```
# Usage

```bash
import gym_act

env = gym.make("Act-v0")

done = False
while not done:
    action = 0 # Your agent code here
    obs, reward, done, _ = env.step(action)
    img = env.render()
```

