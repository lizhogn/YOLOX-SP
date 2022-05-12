exp_file[0]="/home/zhognli/YOLOX/exps/microtubular/mini_data_noaug/mini_data20.py"
exp_file[1]="/home/zhognli/YOLOX/exps/microtubular/mini_data_noaug/mini_data30.py"
exp_file[2]="/home/zhognli/YOLOX/exps/microtubular/mini_data_noaug/mini_data40.py"
exp_file[3]="/home/zhognli/YOLOX/exps/microtubular/mini_data_noaug/mini_data50.py"
exp_file[4]="/home/zhognli/YOLOX/exps/microtubular/mini_data_noaug/mini_data75.py"
exp_file[5]="/home/zhognli/YOLOX/exps/microtubular/mini_data_noaug/mini_data100.py"

expn=("mini_data20_noaug" "mini_data30_noaug" "mini_data40_noaug" "mini_data50_noaug" "mini_data75_noaug" "mini_data100_noaug")


for loop in 0 1 2 3 4
do
    echo "python tools/eval.py --exp_file ${exp_file[loop]} --experiment-name ${expn[loop]} --ckpt /home/zhognli/YOLOX/exps/${expn[loop]}/latest_ckpt.pth"
    python tools/eval.py --exp_file ${exp_file[loop]} --experiment-name ${expn[loop]} --ckpt /home/zhognli/YOLOX/YOLOX_outputs/${expn[loop]}/latest_ckpt.pth
done