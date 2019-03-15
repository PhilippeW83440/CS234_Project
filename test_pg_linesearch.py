import numpy as np
from numpy import linalg as LA
import tensorflow as tf
import gym
import time
import spinup.algos.vpg.core as core
import spinup.algos.trpo.core as trpo_core
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_tf import MpiAdamOptimizer, sync_all_params
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
import sys; sys.path.append("/home/vaishali/CS234_Project/gym-act")
import gym_act
from scipy import optimize


env='Act-v0'
hid=300
l=1
gamma=0.99
seed=0
epochs=100
steps=100
EPS = 1e-8
exp_name='pg_linesearch'

from spinup.utils.run_utils import setup_logger_kwargs
logger_kwargs = setup_logger_kwargs(exp_name, seed)



import numpy as np
import tensorflow as tf
import gym
import time
import spinup.algos.vpg.core as core
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_tf import MpiAdamOptimizer, sync_all_params
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs


class PGBuffer:
    """
    A buffer for storing trajectories experienced by a VPG agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        return [self.obs_buf, self.act_buf, self.adv_buf, 
                self.ret_buf, self.logp_buf]


"""

Vanilla Policy Gradient

(with GAE-Lambda for advantage estimation)

"""
def pg_linesearch(env_fn, actor_critic=core.mlp_actor_critic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=4000, epochs=50, gamma=0.99, pi_lr=3e-4, 
        backtrack_coeff=0.8,delta=0.01, backtrack_iters=1000,
        vf_lr=1e-3, train_v_iters=80, lam=0.97, max_ep_len=1000,
        logger_kwargs=dict(), save_freq=10):
    """

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: A function which takes in placeholder symbols 
            for state, ``x_ph``, and action, ``a_ph``, and returns the main 
            outputs from the agent's Tensorflow computation graph:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       (batch, act_dim)  | Samples actions from policy given 
                                           | states.
            ``logp``     (batch,)          | Gives log probability, according to
                                           | the policy, of taking actions ``a_ph``
                                           | in states ``x_ph``.
            ``logp_pi``  (batch,)          | Gives log probability, according to
                                           | the policy, of the action sampled by
                                           | ``pi``.
            ``v``        (batch,)          | Gives the value estimate for states
                                           | in ``x_ph``. (Critical: make sure 
                                           | to flatten this!)
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the actor_critic 
            function you provided to VPG.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.

        gamma (float): Discount factor. (Always between 0 and 1.)

        pi_lr (float): Learning rate for policy optimizer.

        vf_lr (float): Learning rate for value function optimizer.

        train_v_iters (int): Number of gradient descent steps to take on 
            value function per epoch.

        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    seed += 10000 * proc_id()
    tf.set_random_seed(seed)
    np.random.seed(seed)

    env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape
    
    # Share information about action space with policy architecture
    ac_kwargs['action_space'] = env.action_space

    # Inputs to computation graph
    x_ph, a_ph = core.placeholders_from_spaces(env.observation_space, env.action_space)
    adv_ph, ret_ph, logp_old_ph = core.placeholders(None, None, None)

    # Main outputs from computation graph
    pi, logp, logp_pi, v = actor_critic(x_ph, a_ph, **ac_kwargs)

    # Need all placeholders in *this* order later (to zip with data from buffer)
    all_phs = [x_ph, a_ph, adv_ph, ret_ph, logp_old_ph]

    # Every step, get: action, value, and logprob
    get_action_ops = [pi, v, logp_pi]

    # Experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buf = PGBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)

    # Count variables
    var_counts = tuple(core.count_vars(scope) for scope in ['pi', 'v'])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)

    # VPG objectives
    pi_loss = -tf.reduce_mean(logp * adv_ph)
    v_loss = tf.reduce_mean((ret_ph - v)**2)

    # Info (useful to watch during learning)
    approx_kl = tf.reduce_mean(logp_old_ph - logp)      # a sample estimate for KL-divergence, easy to compute
    approx_ent = tf.reduce_mean(-logp)                  # a sample estimate for entropy, also easy to compute



    # Optimizers
    #train_pi = MpiAdamOptimizer(learning_rate=pi_lr).minimize(pi_loss)
    train_v = MpiAdamOptimizer(learning_rate=vf_lr).minimize(v_loss)

    # Symbols needed for CG solver
    pi_params = trpo_core.get_vars('pi')
    gradient = trpo_core.flat_grad(pi_loss, pi_params)
    #v_ph, hvp = trpo_core.hessian_vector_product(d_kl, pi_params)
    v_ph = tf.placeholder(tf.float32, shape=gradient.shape)
    ##TODO: more analysis on damping Coeff
    #if damping_coeff > 0:
        #hvp += damping_coeff * v_ph

    # Symbols for getting and setting params
    get_pi_params = trpo_core.flat_concat(pi_params)
    set_pi_params = trpo_core.assign_params_from_flat(v_ph, pi_params)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Sync params across processes
    sess.run(sync_all_params())

    # Setup model saving
    logger.setup_tf_saver(sess, inputs={'x': x_ph}, outputs={'pi': pi, 'v': v})

    def cg(Ax, b):
        """
        Conjugate gradient algorithm
        (see https://en.wikipedia.org/wiki/Conjugate_gradient_method)
        """

        ##TODO: Next Step is to try the hessian 
        x = np.zeros_like(b)
        r = b.copy() # Note: should be 'b - Ax(x)', but for x=0, Ax(x)=0. Change if doing warm start.
        p = r.copy()
        r_dot_old = np.dot(r,r)
        cg_iters = 20
        for _ in range(cg_iters):
            z = Ax(p)
            alpha = r_dot_old / (np.dot(p, z) + EPS)
            x += alpha * p
            r -= alpha * z
            r_dot_new = np.dot(r,r)
            p = r + (r_dot_new / r_dot_old) * p
            r_dot_old = r_dot_new
        return x


    def update():
        inputs = {k:v for k,v in zip(all_phs, buf.get())}
        #TODO: Next step is to calculate the hessian using safe distance
        #Hx = lambda x : mpi_avg(sess.run(hvp, feed_dict={**inputs, v_ph: x}))
        g, pi_l_old, v_l_old, ent = sess.run([gradient, pi_loss, v_loss, approx_ent], feed_dict=inputs)
        g, pi_l_old = mpi_avg(g), mpi_avg(pi_l_old)
        #x = cg(Hx, g)
        #x = optimize.fmin_cg(pi_l_old, x0, fprime=g)
        x = g
        old_params = sess.run(get_pi_params)
        old_penalty = env.penalty(env.s)
        alpha = np.sqrt(2*delta/(np.dot(x, g)+EPS))


        # backtracking line search, hard constraint check on env penalty
        for j in range(backtrack_iters):
                step=backtrack_coeff**j
                sess.run(set_pi_params, feed_dict={v_ph: old_params - alpha * x * step})
                pi_l_new = sess.run([pi_loss], feed_dict=inputs)
                penalty = env.penalty(env.s)
                #print("Old Penalty {}, Penalty {}".format(old_penalty,penalty))

                if penalty == 0 or penalty < old_penalty:
                #if pi_l_new <= pi_l_old:
                    logger.log('Accepting new params at step %d of line search.'%j)
                    logger.store(BacktrackIters=j)
                    logger.store(penalty=penalty, old_penalty=old_penalty)
                    break

                if j==backtrack_iters-1:
                    logger.log('Line search failed! Keeping old params.')
                    logger.store(BacktrackIters=j)
                    logger.store(penalty=penalty, old_penalty=old_penalty)


        # Policy gradient step
        #sess.run(train_pi, feed_dict=inputs)

        # Value function learning
        for _ in range(train_v_iters):
            sess.run(train_v, feed_dict=inputs)


        # Log changes from update
        #pi_l_new, v_l_new, kl = sess.run([pi_loss, v_loss, approx_kl], feed_dict={v_ph: old_params - alpha * x * step})
        logger.store(LossPi=pi_l_old, 
                     Entropy=ent, 
                     DeltaLossPi=(pi_l_new - pi_l_old))



    start_time = time.time()
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        for t in range(local_steps_per_epoch):
            a, v_t, logp_t = sess.run(get_action_ops, feed_dict={x_ph: o.reshape(1,-1)})

            # save and log
            buf.store(o, a, r, v_t, logp_t)
            logger.store(VVals=v_t)

            o, r, d, _ = env.step(a[0])
            ep_ret += r
            ep_len += 1

            terminal = d or (ep_len == max_ep_len)
            if terminal or (t==local_steps_per_epoch-1):
                if not(terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len)
                # if trajectory didn't reach terminal state, bootstrap value target
                last_val = r if d else sess.run(v, feed_dict={x_ph: o.reshape(1,-1)})
                buf.finish_path(last_val)
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet=ep_ret, EpLen=ep_len)
                o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs-1):
            logger.save_state({'env': env}, None)

        # Perform PG update!
        update()

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', average_only=True)
        logger.log_tabular('penalty', average_only=True)
        logger.log_tabular('old_penalty', average_only=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()




pg_linesearch(lambda : gym.make(env), actor_critic=core.mlp_actor_critic, ac_kwargs=dict(hidden_sizes=[hid]*l), gamma=gamma, seed=seed, steps_per_epoch=steps, epochs=epochs,logger_kwargs=logger_kwargs)


