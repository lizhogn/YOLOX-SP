# exp file
exp_file[0]="/home/zhognli/YOLOX/exps/microtubular/mini_data_noaug/mini_data20.py"
exp_file[1]="/home/zhognli/YOLOX/exps/microtubular/mini_data_noaug/mini_data30.py"
exp_file[2]="/home/zhognli/YOLOX/exps/microtubular/mini_data_noaug/mini_data40.py"
exp_file[3]="/home/zhognli/YOLOX/exps/microtubular/mini_data_noaug/mini_data50.py"
exp_file[4]="/home/zhognli/YOLOX/exps/microtubular/mini_data_noaug/mini_data75.py"
exp_file[5]="/home/zhognli/YOLOX/exps/microtubular/mini_data_noaug/mini_data100.py"


batch_size=(2 2 2 2 2 2)
exp_name=("mini_data20_noaug" "mini_data30_noaug" "mini_data40_noaug" "mini_data50_noaug" "mini_data75_noaug" "mini_data100_noaug")

for loop in 0 1 2 3 4 5
do
    # echo "tools/train.py --exp_file ${exp_file[loop]} --batch-size ${batch_size[loop]} --experiment-name ${exp_name[loop]}"
    python tools/train.py --exp_file ${exp_file[loop]} --batch-size ${batch_size[loop]} --experiment-name ${exp_name[loop]}
done

# sh tools/train.py \
# --experiment_name miniset_50 \
# --batch_size 2 \
# --exp_file /home/zhognli/YOLOX/exps/mini_data/mini_data50.py