'''
  Run experiment with wandb logging.

  Usage:
  python runexpwb.py --setting bag

  Note: wandb isn't compatible with running scripts in subdirs:
    e.g., python -m exps.chess.chessgfn
  So we call wandb init here.
'''
import random
import torch
import wandb
import options
import numpy as np
from attrdict import AttrDict

from exps.tfbind8 import tfbind8_oracle
from exps.qm9str import qm9str
from exps.sehstr import sehstr
from exps.rna import rna

setting_calls = {
  'tfbind8': lambda args: tfbind8_oracle.number_of_modes(args),
  'qm9str': lambda args: qm9str.number_of_modes(args),
  'sehstr': lambda args: sehstr.number_of_modes(args),
  'rna': lambda args: rna.number_of_modes(args),
}

def number_of_modes(args):
  print(f'Using {args.setting} ...')
  exp_f = setting_calls[args.setting]
  exp_f(args)
  return

def set_seed(seed=0):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)


if __name__ == '__main__':
  args = options.parse_args()
  
  set_seed(args.seed)
  
  if args.setting == "rna":
    args.saved_models_dir = f"{args.saved_models_dir}/L{args.rna_length}_RNA{args.rna_task}/" 
    wandb.init(project=f"{args.wandb_project}-L{args.rna_length}-{args.rna_task}",
              entity=args.wandb_entity,
              config=args,
              mode=args.wandb_mode)
  else:
    wandb.init(project=args.wandb_project,
              entity=args.wandb_entity,
              config=args, 
              mode=args.wandb_mode)
  args = AttrDict(wandb.config)

  run_name = args.model
  if args.model == 'subtb':
    run_name += f"{args.lamda}"
  
  if args.offline_select == "prt":
    run_name += "_" + args.offline_select
  
  if args.sa_or_ssr == "ssr":
    run_name += "_" + args.sa_or_ssr

  if args.ls:
    run_name += "_" + "ls"
    if args.deterministic:
      run_name += "_" + "deterministic"
    run_name += "_" + f"k{args.k}"
    run_name += "_" + f"i{args.i}"
    
  run_name += "_" + f"beta{args.beta}"
  run_name += "_" + f"seed{args.seed}"
  
  args.run_name = run_name.upper()
  print(f"Save model into {args.run_name}")

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  args.device = device
 
  number_of_modes(args)
