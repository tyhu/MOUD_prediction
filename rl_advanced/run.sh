rm tmp
for i in 0.3 0.2 0.1 0.05 0.01 0 -0.01 -0.05 -0.1
do
    echo $i
    python 03_rl.py $i >>tmp
done
