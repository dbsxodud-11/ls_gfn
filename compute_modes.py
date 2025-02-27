import pickle
import argparse
import numpy as np
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="L14_RNA1")
    args = parser.parse_args()
    
    seq_length = 14
    
    # Load data
    ys = np.load(f"datasets/rna/{args.task}_allpreds.pkl", allow_pickle=True)
    alphabet = ["A", "U", "C", "G"]
    num_classes = len(alphabet)
    
    # Threshold
    print("Finding modes ...")
    mode_perecentile = 0.005
    mode_r_threshold = np.percentile(ys, 100*(1-mode_perecentile))
    
    # Modes with Threshold
    modes = np.arange(ys.shape[0])[ys >= mode_r_threshold].tolist()
    print(f"Found num modes: {len(modes)}")
    
    # Modes with Hamming Ball Constraint
    modes_hammingball1 = set()
    modes_hammingball2 = set()
    with tqdm(total=len(modes)) as pbar:
        for mode in modes:
            mode_y = ys[mode]
            # convert mode to sequence
            mode_idx = []
            for i in range(seq_length):
                idx = mode // num_classes**i % num_classes
                mode_idx.append(idx)
                mode = mode - idx * num_classes**i
            mode_idx = mode_idx[::-1]
            
            flag = True
            # find 1-hamming neighbors
            for i in range(seq_length):
                for j in range(num_classes):
                    if j != mode_idx[i]:
                        neighbor_idx = np.array(mode_idx)
                        neighbor_idx[i] = j
                        neighbor = sum(neighbor_idx[i] * num_classes**(seq_length - i - 1) for i in range(seq_length))
                        if ys[neighbor] > mode_y:
                            flag = False
                            break
            
            if flag:
                modes_hammingball1.add(sum(mode_idx[i] * num_classes**(seq_length - i - 1) for i in range(seq_length)))
                # find 2-hamming neighbors
                for i in range(seq_length):
                    for j in range(i+1, seq_length):
                        for k in range(num_classes):
                            for l in range(num_classes):
                                if k != mode_idx[i] and l != mode_idx[j]:
                                    neighbor_idx = np.array(mode_idx)
                                    neighbor_idx[i] = k
                                    neighbor_idx[j] = l
                                    neighbor = sum(neighbor_idx[i] * num_classes**(seq_length - i - 1) for i in range(seq_length))
                                    if ys[neighbor] > mode_y:
                                        flag = False
                                        break
                if flag:
                    modes_hammingball2.add(sum(mode_idx[i] * num_classes**(seq_length - i - 1) for i in range(seq_length)))
                
            pbar.update(1)
            pbar.set_postfix({"num modes (dist=1)": len(modes_hammingball1), "num modes (dist=2)": len(modes_hammingball2)})
            
        # convert mode to string
        modes_str = set()
        for mode in modes:
            mode_str = ""
            for i in range(seq_length):
                idx = mode // num_classes**i % num_classes
                mode_str += alphabet[idx]
            modes_str.add(mode_str)
        
        modes_hammingball1_str = set()
        for mode in modes_hammingball1:
            mode_str = ""
            for i in range(seq_length):
                idx = mode // num_classes**i % num_classes
                mode_str += alphabet[idx]
            modes_hammingball1_str.add(mode_str)
            
        modes_hammingball2_str = set()
        for mode in modes_hammingball2:
            mode_str = ""
            for i in range(seq_length):
                idx = mode // num_classes**i % num_classes
                mode_str += alphabet[idx]
            modes_hammingball2_str.add(mode_str)

        with open(f"datasets/rna/{args.task}/mode_info.pkl", "wb") as f:
            pickle.dump({"modes": modes_str, "modes_hamming_ball1": modes_hammingball1_str, "modes_hamming_ball2": modes_hammingball2_str}, f)
        