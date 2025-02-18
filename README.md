# OmniSafe

## Installation

```bash
git clone https://github.com/PKU-Alignment/omnisafe.git
cd omnisafe
conda create -n omnisafe python=3.8
conda activate omnisafe
pip install -e .
```

#### Note

- Change the version of `cvxopt` to `1.1.8` in order to install OmniSafe.
- Upgrade decorator, otherwise rendering does not seem to work.
`pip install --upgrade decorator==4.0.2`

## Basic Usage

#### `train_policy.py`
This has the simplest example of training an agent for a given environment.The
configuration is passed using a dictionary but can also be passed using YAML
files.

Running this script creates a `/runs` folder. 

#### `evaluate_policy.py` 
For a given trained policy we can evaluate the results by providing the location
of the logs directory.

#### `experiment_grid.py`
In order to train combinations of multiple policies on differnet environments
we can use the `ExperimentGrid` module. It takes a list of policies and
environments.

#### `env_from_scratch.py`
Create an env for safe reinforcement learning and register it with OmniSafe.
These environments have an aditional cost component along with the regular
reward signal.

#### `embed_env.py`
Embed an existing gym-style environment into one which will work with OmniSafe.

## Environments
```
'SafetyRacecarButton2-v0', 'SafetyDoggoButton2-v0', 'SafetyPointCircle2-v0',
'SafetyPointButton2-v0', 'SafetyCarRun0-v0', 'SafetyAntPush0-v0', 'Humanoid-v4',
'SafetyAntGoal0-v0-modelbased', 'SafetyAntGoal2-v0', 'SafetyCarButton1-v0',
'SafetySwimmerVelocity-v1', 'SafetyRacecarPush0-v0',
'ShadowHandCatchOver2UnderarmSafeJoint', 'ShadowHandOverSafeJoint',
'SafetyAntCircle1-v0', 'SafetyAntCircle0-v0', 'SafetyPointRun0-v0',
'SafetyDoggoPush0-v0', 'SafetyCarGoal0-v0-modelbased', 'SafetyDoggoButton0-v0',
'SafetyRacecarPush1-v0', 'SafetyDoggoGoal2-v0', 'SafetyRacecarGoal0-v0',
'SafetyRacecarButton1-v0', 'SafetyCarGoal1-v0-modelbased',
'SafetyDoggoCircle0-v0', 'SafetyCarGoal2-v0', 'SafetyAntVelocity-v1',
'SafetyCarButton2-v0', 'SafetyPointGoal1-v0-modelbased', 'SafetyPointPush1-v0',
'SafetyCarGoal1-v0', 'SafetyPointGoal0-v0-modelbased', 'SafetyRacecarGoal2-v0',
'SafetyAntButton2-v0', 'SafetyCarButton0-v0', 'SafetyCarPush0-v0',
'SafetyPointButton1-v0', 'SafetyAntCircle2-v0', 'SafeInvertedPendulumSwing-v2',
'SafetyHopperVelocity-v1', 'SafetyAntGoal0-v0', 'SafetyCarCircle1-v0',
'SafetyPointButton0-v0', 'SafetyDoggoCircle2-v0', 'SafetyPointGoal1-v0',
'SafetyHalfCheetahVelocity-v1', 'SafetyCarCircle2-v0', 'SafetyDoggoGoal1-v0',
'Swimmer-v4', 'SafetyCarGoal0-v0', 'Simple-v0', 'SafetyRacecarCircle1-v0',
'Hopper-v4', 'Walker2d-v4', 'ShadowHandCatchOver2UnderarmSafeFinger',
'SafetyAntPush1-v0', 'SafetyRacecarGoal1-v0', 'SafetyPointPush2-v0',
'SafetyPointPush0-v0', 'SafetyPointGoal2-v0', 'SafetyAntGoal1-v0',
'SafetyDoggoButton1-v0', 'SafetyAntGoal1-v0-modelbased', 'SafetyCarPush2-v0',
'SafetyDoggoPush2-v0', 'HalfCheetah-v4', 'SafetyAntButton0-v0',
'SafetyCarPush1-v0', 'SafetyCarCircle0-v0', 'SafetyDoggoCircle1-v0',
'SafetyDoggoGoal0-v0', 'SafetyRacecarCircle0-v0', 'SafeMetaDrive',
'SafetyWalker2dVelocity-v1', 'SafetyAntPush2-v0', 'SafetyPointCircle0-v0',
'SafetyPointGoal0-v0', 'SafetyDoggoPush1-v0', 'SafetyRacecarCircle2-v0',
'SafetyHumanoidVelocity-v1', 'SafetyAntButton1-v0', 'SafetyRacecarPush2-v0',
'SafetyRacecarButton0-v0', 'SafetyPointCircle1-v0', 'ShadowHandOverSafeFinger',
'Ant-v4'
```

## Agents (Algorithms)

The agents mentioned below can be benchedmarked by using the `ExperimentGrid`
module and adding the environemnt using `eg.add("algo", [PPO, PPOLag, P30, ...])`

#### On-Policy Algorithms

[Docs page for On-Policy Algorithms details](https://www.omnisafe.ai/en/latest/benchmark/on-policy.html)

**First-Order**

- **[NIPS 1999]** [Policy Gradient (PG)](https://papers.nips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf)
- **[Preprint 2017]** [Proximal Policy Optimization (PPO)](https://arxiv.org/pdf/1707.06347.pdf)
- [The Lagrange version of PPO (PPOLag)](https://cdn.openai.com/safexp-short.pdf)
- **[IJCAI 2022]** [Penalized Proximal Policy Optimization for Safe Reinforcement Learning (P3O)]( https://arxiv.org/pdf/2205.11814.pdf)
- **[NeurIPS 2020]** [First Order Constrained Optimization in Policy Space (FOCOPS)](https://arxiv.org/abs/2002.06506)
- **[NeurIPS 2022]**  [Constrained Update Projection Approach to Safe Policy Optimization (CUP)](https://arxiv.org/abs/2209.07089)

**Second-Order**

- **[NeurIPS 2001]** [A Natural Policy Gradient (NaturalPG))](https://proceedings.neurips.cc/paper/2001/file/4b86abe48d358ecf194c56c69108433e-Paper.pdf)
- **[PMLR 2015]** [Trust Region Policy Optimization (TRPO)](https://arxiv.org/abs/1502.05477)
- [The Lagrange version of TRPO (TRPOLag)](https://cdn.openai.com/safexp-short.pdf)
- **[ICML 2017]** [Constrained Policy Optimization (CPO)](https://proceedings.mlr.press/v70/achiam17a)
- **[ICML 2017]** [Proximal Constrained Policy Optimization (PCPO)](https://proceedings.mlr.press/v70/achiam17a)
- **[ICLR 2019]** [Reward Constrained Policy Optimization (RCPO)](https://openreview.net/forum?id=SkfrvsA9FX)

**Saute RL**

- **[ICML 2022]** [Sauté RL: Almost Surely Safe Reinforcement Learning Using State Augmentation (PPOSaute, TRPOSaute)](https://arxiv.org/abs/2202.06558)

**Simmer**

- **[NeurIPS 2022]** [Effects of Safety State Augmentation on Safe Exploration (PPOSimmerPID, TRPOSimmerPID)](https://arxiv.org/abs/2206.02675)

**PID-Lagrangian**

- **[ICML 2020]** [Responsive Safety in Reinforcement Learning by PID Lagrangian Methods (CPPOPID, TRPOPID)](https://arxiv.org/abs/2007.03964)

**Early Terminated MDP**

- **[Preprint 2021]** [Safe Exploration by Solving Early Terminated MDP (PPOEarlyTerminated, TRPOEarlyTerminated)](https://arxiv.org/pdf/2107.04200.pdf)

#### Model-based Algorithms

[Docs page for Model Based Algorithms details](https://www.omnisafe.ai/en/latest/benchmark/modelbased.html)

- **[NeurIPS 2001]** [Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models (PETS))](https://arxiv.org/abs/1805.12114)
- **[CoRL 2021]** [Learning Off-Policy with Online Planning (LOOP and SafeLOOP)](https://arxiv.org/abs/2008.10066)
- **[AAAI 2022]** [Conservative and Adaptive Penalty for Model-Based Safe Reinforcement Learning (CAP)](https://arxiv.org/abs/2112.07701)
- **[ICML 2022 Workshop]** [Constrained Model-based Reinforcement Learning with Robust Cross-Entropy Method (RCE)](https://arxiv.org/abs/2010.07968)
- **[NeurIPS 2018]** [Constrained Cross-Entropy Method for Safe Reinforcement Learning (CCE)](https://proceedings.neurips.cc/paper/2018/hash/34ffeb359a192eb8174b6854643cc046-Abstract.html)

## Custom Agent (Notes)

#### `/sac`
This folder has an implementaion for the Soft Actor Critic algorithm meant to
eventually registered with OpenSafe.

#### OpenSafe working
- What happens when we call `.learn()` on an `Agent` object?
  - `AlgoWrapper` is imported as `Agent`. This uses the registry to create object of the algorithm class (of type `BaseAlgo`)
  - `BaseAlgo` has the `.learn()` method. This in turn calls the `.learn()` of the actual algorithm class.  
  - What exactly happens inside the `.learn()` method of the `BaseAlgo` class object?
    - rollout?
    - update?
    - log?

- What happens when we call the `.render()` method of a `BaseAlgo` object?
  - This uses the `Evaluator` to load a model and render.
  
- What does an `Evaluator` look for when we call `.load_saved()`?
  - ...

- What happens when we call `.evaluate` on a `Evaluator` object?
  - ...
