'''
  Run experiment with wandb logging.

  Usage:
  python main.py --setting qm9str

  Note: wandb isn't compatible with running scripts in subdirs:
    e.g., python -m exps.chess.chessgfn
  So we call wandb init here.
'''

import argparse
import random
import torch
import wandb
import options
import numpy as np
from attrdict import AttrDict

from exps.qm9str import qm9str
from exps.sehstr import sehstr
from exps.tfbind8 import tfbind8_oracle
from exps.rna import rna

mode_seeking = {
    'qm9str': lambda args: qm9str.mode_seeking(args),
    'sehstr': lambda args: sehstr.mode_seeking(args),
    'tfbind8': lambda args: tfbind8_oracle.mode_seeking(args),
    'rna1': lambda args: rna.mode_seeking(args),
    'rna2': lambda args: rna.mode_seeking(args),
    'rna3': lambda args: rna.mode_seeking(args),
}

def main(args):
    print(f"Setting: {args.setting}")
    exp_f = mode_seeking[args.setting]
    exp_f(args)
    return

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    

if __name__ == "__main__":
    args = options.parse_args()
    set_seed(args.seed)

    wandb.init(project=args.wandb_project,
               entity=args.wandb_entity,
               config=args, 
               mode=args.wandb_mode)
    args = AttrDict(wandb.config)
    
    if args.setting.startswith("rna"):
        args.saved_models_dir += args.setting + "/"
    
    if args.model == "gfn":
        if args.ls:
            run_name = "ls_gfn"
            run_name += "_" + args.filtering
            run_name += "_" + f"i{args.num_iterations}"
            run_name += "_" + f"k{args.num_back_forth_steps}"
        else:
            run_name = "gfn"
            
        run_name += "_" + args.loss_type
        if args.loss_type == "subtb":
            run_name += f"{float(args.lamda)}"
    else:
        run_name = args.model
    run_name += "/"
    
    run_name += f"seed{args.seed}"
    args.run_name = run_name.upper()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device={device}')
    args.device = device
    
    main(args)
