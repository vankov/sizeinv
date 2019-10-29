#!/bin/bash   

scale=12
max_scale=11
min_scale=2
min_d=1
max_d=11

for i in {1..100}
do
    echo "Running simulation #$i"
    for (( t=$min_scale; t<=$max_scale; t += 2 ))
    do
        for (( d=$min_scale; d<=$max_scale; d += 2 ))
        do    
            max=$((t + d))
            min=$((t - d))
            
            if (($min >= $min_d)) || (($max <= $max_d))
            then
                echo $t $d            
                p make.py -t $t -d $d -noise 0 -n 1 -min_train_size $min_scale -max_train_size $max_scale -image_scale $scale
                titanX p train.py -t=$t -d=$d -noise=0 -gap -pretrained -stop_patience 2 -stop_min_delta 0.01 -epochs 10 -train_steps 2400 >> results/$t.$d.0.1.txt            
            fi
        done
    done    
done