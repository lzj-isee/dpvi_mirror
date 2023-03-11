"""
MSVGDd method in dual space
"""
import os
command = 'python main.py --config_path ./configs/default.yaml --over_writen --algorithm MSVGDd  --task dirichletSingle --particle_num 256 --lr 0.2 --alpha 0.02 --knType rbf --bwType med  --bwVal 0.1 --max_iter 5000 --max_time 120 --optim sgd --optim_rho 0.9 --reference_num 2000 --eval_interval 100 --device cuda:3 --result_dir results --seed 0'
os.system(command)


"""
MSVGDp method in primal space
"""
import os
command = 'python main.py --config_path ./configs/default.yaml --over_writen --algorithm MSVGDp  --task dirichletSingle --particle_num 256 --lr 0.2 --alpha 0.02 --knType rbf --bwType med  --bwVal 0.1 --max_iter 5000 --max_time 120 --optim sgd --optim_rho 0.9 --reference_num 2000 --eval_interval 100 --device cuda:3 --result_dir results --seed 0'
os.system(command)


"""
MedBLOBd method in dual space
"""
import os
command = 'python main.py --config_path ./configs/default.yaml --over_writen --algorithm MedBLOBd  --task dirichletSingle --particle_num 256 --lr 0.01 --alpha 0.02 --knType rbf --bwType nei  --bwVal 0.1 --max_iter 5000 --max_time 120 --optim sgd --optim_rho 0.9 --reference_num 2000 --eval_interval 100 --device cuda:3 --result_dir results --seed 0'
os.system(command)


"""
MedBLOBdCA method in dual space
"""
import os
command = 'python main.py --config_path ./configs/default.yaml --over_writen --algorithm MedBLOBdCA  --task dirichletSingle --particle_num 256 --lr 0.01 --alpha 0.1 --knType rbf --bwType nei  --bwVal 0.1 --max_iter 5000 --max_time 120 --optim sgd --optim_rho 0.9 --reference_num 2000 --eval_interval 100 --device cuda:3 --result_dir results --seed 0'
os.system(command)


"""
MedBLOBp method in primal space
"""
import os
command = 'python main.py --config_path ./configs/default.yaml --over_writen --algorithm MedBLOBp  --task dirichletSingle --particle_num 256 --lr 0.002 --alpha 0.02 --knType rbf --bwType nei  --bwVal 0.1 --max_iter 5000 --max_time 120 --optim sgd --optim_rho 0.9 --reference_num 2000 --eval_interval 100 --device cuda:3 --result_dir results --seed 0'
os.system(command)


"""
MedBLOBpCA method in primal space
"""
import os
command = 'python main.py --config_path ./configs/default.yaml --over_writen --algorithm MedBLOBpCA  --task dirichletSingle --particle_num 256 --lr 0.002 --alpha 0.2 --knType rbf --bwType nei  --bwVal 0.1 --max_iter 5000 --max_time 120 --optim sgd --optim_rho 0.9 --reference_num 2000 --eval_interval 100 --device cuda:3 --result_dir results --seed 0'
os.system(command)