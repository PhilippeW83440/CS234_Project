# CS234_Project  

**Project install**  

```bash
git clone https://github.com/PhilippeW83440/CS234_Project.git
cd CS234_Project
pip install -r requirements.txt
cd gym-act
pip install -e .
```
Then check you can run the notebooks Tests_xxx.ipynb without error.  
Mujoco is not required, so just disable Mujoco tests if you do not have a license.  


**Spinning up**  
https://blog.openai.com/spinning-up-in-deep-rl/  
https://spinningup.openai.com/en/latest/  
https://github.com/openai/spinningup  
https://blog.openai.com/concrete-ai-safety-problems/  
  
https://github.com/yanpanlau/DDPG-Keras-Torcs  
https://medium.com/@scitator/run-skeleton-run-3rd-place-solution-for-nips-2017-learning-to-run-207f9cc341f8  
 
 
 **Implementing a custom gradient descent in Tensorflow**  
 https://stackoverflow.com/questions/39167070/implementing-gradient-descent-in-tensorflow-instead-of-using-the-one-provided-wi  
 https://towardsdatascience.com/custom-optimizer-in-tensorflow-d5b41f75644a  

  
**Safe RL overview:**
* A Comprehensive Survey on Safe Reinforcement Learning  
  http://www.jmlr.org/papers/volume16/garcia15a/garcia15a.pdf
* https://las.inf.ethz.ch/files/ewrl18_SafeRL_tutorial.pdf
* https://www.youtube.com/watch?v=saHMbn84V_s
* https://medium.com/@harshitsikchi/towards-safe-reinforcement-learning-88b7caa5702e

**Safe RL papers:**
* Safe Control under Uncertainty, 2015 D. Sadigh (Berkeley), Kapoor (Microsoft Research)  
  https://arxiv.org/abs/1510.07313  
* Safe Exploration in Continuous Action Spaces, 2018, Dalal + DeepMind  
  https://arxiv.org/abs/1801.08757  
* Constrained Policy Optimization, 2017, J Achiam,  P Abbeel (Berkeley)   
  https://arxiv.org/abs/1705.10528  
* Uncertainty-Aware Reinforcement Learning for Collision Avoidance, G. Kahn, , P Abbeel (Berkeley)  
  https://arxiv.org/abs/1702.01182  
* Safe Actor-Critic, 2018 MacGill + DeepMind  
  https://sites.google.com/view/rl-uai2018/schedule  
* Design of safe control policies for large-scale non-linear systems operating in uncertain environments  (INRIA, Renault)    
  https://eleurent.github.io/robust-control/  
* Reinforcement learning under circumstances beyond its control  
  https://pdfs.semanticscholar.org/7c61/0c97c56e9e3108af9ac00faacce5dbd2ac0c.pdf  
* Reward Constrained Policy Optimization, 2019, Technion + DeepMind  
  https://arxiv.org/abs/1805.11074  
* Value constrained model-free continuous control, 2018, DeepMind  
  http://phys2018.csail.mit.edu/papers/51.pdf  

Useful tools: 
* dual gradient ascent algorithm: https://ee227c.github.io/notes/ee227c-lecture14.pdf  
  
  
**RL and AD:**  
* Reinforcement Learning for Autonomous Maneuvering in Highway Scenarios, 2017, Werling (BMW)   
  https://www.uni-das.de/images/pdf/veroeffentlichungen/2017/04.pdf
* High-level Decision Making for Safe and Reasonable Autonomous Lane Changing using Reinforcement Learning, 2018 Werling (BMW)  
  http://mediatum.ub.tum.de/doc/1454224/712763187208.pdf  
* Exploring applications of deep reinforcement learning for real-world autonomous driving systems, 2019, ENSTA  
  https://arxiv.org/abs/1901.01536  
* Reinforcement Learning with Probabilistic Guarantees for Autonomous Driving, 2018, M. Bouton (Stanford)  
  https://sites.google.com/view/rl-uai2018/schedule  
    
 
**Tensorflow pointers for constrained optimization:**    
* https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/constrained_optimization    
* https://github.com/google-research/tensorflow_constrained_optimization  
