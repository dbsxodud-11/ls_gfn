setting=sehstr
beta=6
k=3
i=7
num_active_learning_rounds=2000


for seed in {0..2}
do
    python runexpwb.py --setting $setting --beta $beta --model tb --num_active_learning_rounds $num_active_learning_rounds --seed $seed &
    python runexpwb.py --setting $setting --beta $beta --model tb --ls true --k $k --i $i --num_active_learning_rounds $num_active_learning_rounds --seed $seed &
    python runexpwb.py --setting $setting --beta $beta --model tb --ls true --deterministic true --k $k --i $i --num_active_learning_rounds $num_active_learning_rounds --seed $seed &
    python runexpwb.py --setting $setting --beta $beta --model a2c --num_active_learning_rounds $num_active_learning_rounds --seed $seed &
    python runexpwb.py --setting $setting --beta $beta --model ppo --num_active_learning_rounds $num_active_learning_rounds --seed $seed &
    python runexpwb.py --setting $setting --beta $beta --model sql --num_active_learning_rounds $num_active_learning_rounds --seed $seed &
    python runexpwb.py --setting $setting --beta $beta --model mars --num_active_learning_rounds $num_active_learning_rounds --seed $seed &
    wait
done
