setting=rna
beta=8
k=7
i=7
mode_metric=hamming_ball1
num_active_learning_rounds=5000

for rna_task in {1..3}
do
    for seed in {0..2}
    do
        python runexpwb.py --setting $setting --rna_task $rna_task --beta $beta --model tb --mode_metric $mode_metric --num_active_learning_rounds $num_active_learning_rounds --seed $seed &
        python runexpwb.py --setting $setting --rna_task $rna_task --beta $beta --model tb --mode_metric $mode_metric --ls true --k $k --i $i --num_active_learning_rounds $num_active_learning_rounds --seed $seed &
        python runexpwb.py --setting $setting --rna_task $rna_task --beta $beta --model tb --mode_metric $mode_metric --ls true --deterministic true --k $k --i $i --num_active_learning_rounds $num_active_learning_rounds --seed $seed &
        python runexpwb.py --setting $setting --rna_task $rna_task --beta $beta --model a2c --mode_metric $mode_metric --num_active_learning_rounds $num_active_learning_rounds --seed $seed &
        python runexpwb.py --setting $setting --rna_task $rna_task --beta $beta --model ppo --mode_metric $mode_metric --num_active_learning_rounds $num_active_learning_rounds --seed $seed &
        python runexpwb.py --setting $setting --rna_task $rna_task --beta $beta --model sql --mode_metric $mode_metric --num_active_learning_rounds $num_active_learning_rounds --seed $seed &
        python runexpwb.py --setting $setting --rna_task $rna_task --beta $beta --model mars --mode_metric $mode_metric --num_active_learning_rounds $num_active_learning_rounds --seed $seed &
        wait
    done
done
