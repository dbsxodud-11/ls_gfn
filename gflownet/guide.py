import copy
import numpy as np
from scipy.special import logsumexp
from collections import namedtuple
from tqdm import tqdm
import ray
import psutil, gc
from attrdict import AttrDict
import queue
from dataclasses import dataclass

from .data import Experience, make_full_exp

RAY_MAX_CALLS = 10
GARBAGE_COLLECT_PCT = 50.0  # [0, 100]
MAX_ACTIVE_JOBS = 96


@dataclass
class Job:
  payload: object
  jobtype: str


""" Guide trajectory sampling (remote ray worker functions)
    Forward : Sample a trajectory ending in a particular x under guide
    Non-guided trajectory sampling is in basegfn.py 
"""
@ray.remote(max_calls = RAY_MAX_CALLS)
def guide_sample(tools, x, allXtoR):
  """ Samples a trajectory ending in x, using substructures shared 
      in allXtoR. Returns Experience.

      Parameters
      ----------
      tools: Ray tools, shared by parent process.
      x: State
      allXtoR: dict, mapping leaf states to rewards.
               Contains all observed training data so far.

      Returns Experience with fields
      -------
      traj: List of States
      x: State
      r: Reward, float
      logp_guide: float, logp(traj) under guide
  """
  allX = copy.copy(set(allXtoR))
  if x in allX:
    allX.remove(x)
  
  traj = [tools.mdp_f.root()]
  logp_guide = 0
  filtered_X = allX
  while traj[-1] != x:
    children, probs = tools.child_p(tools, traj[-1], x, filtered_X, allXtoR)
    node = np.random.choice(children, p=probs)
    logp_guide += np.log(probs[children.index(node)])

    """
      Filter X - suppose some x is not downstream of our current node.
      Then that x cannot be downstream of any of the node's children,
      so it doesn't need to be considered for downstream scoring.
    """
    filtered_X = [item for item in filtered_X
                  if tools.mdp_f.is_member(node, item)]
    traj.append(node)
  return Experience(traj=traj, x=x, r=allXtoR[x], logp_guide=logp_guide)


@ray.remote(max_calls = RAY_MAX_CALLS)
def guide_logp(tools, traj, allXtoR):
  """ Computes logp(traj) under guide. Returns Experience. """
  logp_guide = 0
  x = traj[-1]
  filtered_X = set(allXtoR)
  if x in filtered_X:
    filtered_X.remove(x)
  for i in range(len(traj) - 1):
    children, ps = tools.child_p(tools, traj[i], x, filtered_X, allXtoR)

    # if traj[i+1] not in children:
    #   print(f'Error: State {traj[i+1]=} not in children {children=} of {traj[i]=}. Have {x=}.\n {traj=}')

    logp_guide += np.log(ps[children.index(traj[i+1])])
    filtered_X = [x for x in filtered_X if tools.mdp_f.is_member(traj[i+1], x)]
  return Experience(traj=traj, x=x, r=allXtoR[x], logp_guide=logp_guide)


"""
  Substructure guide
"""
def get_unique_children_in_x(state, x, get_unique_children):
  return [child for child in get_unique_children(state) if child.is_member(x)]


def unique_children_in_x_guide_score(state, 
                                     x, 
                                     X, 
                                     allXtoR,
                                     get_unique_children):
  valid_children = get_unique_children_in_x(state, x, get_unique_children)
  scores = []
  for child in valid_children:
    rs = [allXtoR[_x] for _x in X if child.is_member(_x)]
    scores.append(substruct_scoreagg(rs))
  return valid_children, scores


def substruct_scoreagg(reward_list):
  """ Compute aggregated score for substructure guide.
  
      From a list of rewards for leaf states containing a given state.
      Assumes X are pre-filtered to be downstream of some node.
  """
  if len(reward_list) == 0:
    return 0
  return float(np.mean(reward_list))


# Forward : Sample child of node
def probs_substructure(tools, node, x, X, allXtoR):
  """ Compute prob of children of node, under substructure guide.
  
      Consider children of node in x, and score using X.
      If no children are in X, then choose among children in x.
  """
  valid_children, scores = tools.mdp_f.unique_children_in_x_guide_score(node,
      x, X, allXtoR, tools.mdp_f.get_unique_children)
  if sum(scores) > 0:
    scores = np.maximum(scores, 1e-5)
    log_scores = np.log(scores) * tools.args.guide_sampling_temperature
    scores = np.exp(log_scores - logsumexp(log_scores))
    scores_norm = np.array(scores) / sum(scores)
  else:
    scores_norm = np.ones(len(valid_children)) / len(valid_children)
  return valid_children, scores_norm


def probs_uniform(tools, node, x, X, allXtoR):
  valid_children = tools.mdp_f.get_unique_children_in_x(node, x,
      tools.mdp_f.get_unique_children)
  return valid_children, [1/len(valid_children)]*len(valid_children)


"""
  Ray process manager
"""
fields = ['mdp_f', 'args', 'child_p']
Tools = namedtuple('Tools', fields, defaults=(None,)*len(fields))

class RayManager():
  """ Root thread's manager of Ray parallel jobs, used for 
      computing substructure guide and sampling substructure-aware trajectories. 

      Only stores minimal necessary mdp functions in ray tools, 
      in mdp_f.
  """
  def __init__(self, args, mdp):
    self.args = args
    self.mdp_f = AttrDict({
      'root': mdp.root,
      'is_member': mdp.is_member,
      'get_unique_children': mdp.get_unique_children,
      'get_unique_children_in_x': get_unique_children_in_x,
      'unique_children_in_x_guide_score': unique_children_in_x_guide_score,
    })

    self.ray_tools = self.make_tools()
    self.allXtoR = dict()
    self.ray_xtor = ray.put(self.allXtoR)
    self.futures = []

    if args.model == 'sub' and args.guide != 'substructure':
      print(f'WARNING: Substructure GFN model chosen, but guide is not substructure')
    
    self.job_stack = queue.LifoQueue()
    self.results_qu = queue.Queue()

  def make_tools(self):
    """ Store tools for Ray child processes to access. 
    
        Storage
        -------
        mdp: mdp class object.
          Used functions are: (search in this file for tools.mdp_f.)
            root()
            is_member()
            unique_children_in_x_guide_score()
            get_unique_children_in_x()
        args: arguments
        child_p: Function, assigns probs to children.
          Substructure guided, or uniform.
    """
    print(f'Putting ray tools ...')
    if self.args.guide == 'substructure':
      child_p_f = probs_substructure
    else:
      child_p_f = probs_uniform
    return ray.put(Tools(
      mdp_f = self.mdp_f,
      args = self.args,
      child_p = child_p_f,
    ))

  def update_allXtoR(self, allXtoR):
    del self.ray_xtor
    self.allXtoR = allXtoR
    self.ray_xtor = ray.put(allXtoR)

  """
    Submit jobs - called by trainer
  """
  def submit_online_jobs(self, online_xs):
    """ Sample new trajectories for x with substructure guide.
        Puts onto job_stack. Culls stack if too large. 
    """
    for x in online_xs:
      self.job_stack.put(Job(x, 'guide_traj_sample'))
    self.cull_job_stack()
    self.__update()
    return

  def submit_offline_jobs(self, offline_trajs):
    """ Recompute logp(traj) of replay trajs when X is updated.
        Puts onto offline_stack.  Culls stack if too large.
    """
    for traj in offline_trajs:
      self.job_stack.put(Job(traj, 'recompute_guide_logp'))
    self.cull_job_stack()
    self.__update()
    return

  def __update(self):
    """ Collects all finished jobs, puts in results store,
        starts new jobs if jobs are available on stack.
        Call this frequently. 
    """
    done, notdone = ray.wait(self.futures,
                             num_returns=len(self.futures),
                             timeout=0)
    self.futures = notdone
    results = ray.get(done)
    exps = [make_full_exp(min_exp, self.args) for min_exp in results]
    for exp in exps:
      self.results_qu.put(exp)

    # Start new jobs
    while not self.job_stack.empty() and len(self.futures) < MAX_ACTIVE_JOBS: 
      job = self.job_stack.get()

      if job.jobtype == 'guide_traj_sample':
        x = job.payload
        future = guide_sample.remote(self.ray_tools, x, self.ray_xtor)
      elif job.jobtype == 'recompute_guide_logp':
        traj = job.payload
        future = guide_logp.remote(self.ray_tools, traj, self.ray_xtor)

      self.futures.append(future)

    del results
    garbage_collect()
    print(f'RayManager - in jobs {self.job_stack.qsize()}; results {self.results_qu.qsize()}; active tasks {len(self.futures)}')
    return

  def get_results(self, batch_size=16):
    """ Pop from results store, if batch_size are done. """
    if self.results_qu.qsize() < batch_size:
      return None

    exps = []
    while not self.results_qu.empty() and len(exps) < batch_size:
      exp = self.results_qu.get()
      exps.append(exp)

    self.cull_results_qu()
    self.__update()
    return exps
  
  """
    Culling
  """
  def cull_job_stack(self):
    """ Remove jobs from job stack if it's getting too large. """
    MAX_JOB_STACK_SIZE = 1000
    JOB_STACK_SHRINK_SIZE = 500
    if self.job_stack.qsize() > MAX_JOB_STACK_SIZE:
      new_stack = queue.LifoQueue()
      for _ in range(JOB_STACK_SHRINK_SIZE):
        new_stack.put(self.job_stack.get())
      self.job_stack = new_stack
    return

  def cull_results_qu(self):
    """ Remove from results queue if it's getting too large. """
    MAX_RESULTS_QU_SIZE = 1000
    RESULTS_QU_SHRINK_SIZE = 500    
    if self.results_qu.qsize() > MAX_RESULTS_QU_SIZE:
      for _ in range(MAX_RESULTS_QU_SIZE - RESULTS_QU_SHRINK_SIZE):
        self.results_qu.get()
    return


def garbage_collect(pct=GARBAGE_COLLECT_PCT):
  """ Garbage collect to handle ray memory usage.
      https://stackoverflow.com/questions/55749394/how-to-fix-the-constantly-growing-memory-usage-of-ray
  """
  # print(f'vmem={psutil.virtual_memory().percent}')
  if psutil.virtual_memory().percent >= pct:
    gc.collect()
  return


