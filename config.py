import json
import argparse
import socket



class Config():
    """
    """

    def __init__( **params):
        """
        Extract the parameters, or set them to a default value
        """
        parser = argparse.ArgumentParser(description='Experoiment')
        parser.add_argument('--experiment_name', type=str, default= 'experiment_1')
        parser.add_argument("--path"            , type=str, default=params['base_path']+"/shapenet_tf/")
        parser.add_argument("--checkpoint_path" , type=str, default=params['base_path']+"/shapenet_data/cp")
        parser.add_argument("--saved_model_path", type=str, default=params['base_path']+"/shapenet_data/models")

        parser.add_argument('--model_params_path', type=str, default= './archs/resnet_branch_tanh2.json')
        parser.add_argument('--padding', type=str, default= 'VALID')
        parser.add_argument('--visualize', type=bool, default= False)
        parser.add_argument('--model_params', type=str, default= None)
        parser.add_argument('--batch_size', type=int,  default=32)
        parser.add_argument('--beta1', type=float,  default=0.9)
        parser.add_argument('--dropout', type=float,  default=1.0)
        parser.add_argument('--stage', type=int,  default=0)
        parser.add_argument('--norm_loss_alpha', type=float,  default=0.0000)
        parser.add_argument('--alpha', type=float,  default=0.003)
        parser.add_argument('--grid_size', type=int,  default=36)
        parser.add_argument('--grid_size_v', type=int,  default=256)
        parser.add_argument('--compression', type=int,  default=0)
        parser.add_argument('--pretrained', type=int,  default=0)
        parser.add_argument('--embedding_size', type=int,  default=256)
        parser.add_argument('--num_blocks', type=int,  default=4)
        parser.add_argument('--block_width', type=int,  default=512)
        parser.add_argument('--bottleneck', type=int,  default=512)
        parser.add_argument('--img_size', type=int,  default=[137,137])
        parser.add_argument('--im_per_obj', type=int,  default=24)
        parser.add_argument('--test_size', type=int,  default=24)
        parser.add_argument('--shuffle_size', type=int,  default=1000)  
        parser.add_argument('--test_every', type=int,  default=10000)    
        parser.add_argument('--save_every', type=int,  default=1000) 
        parser.add_argument("--postfix_load"   , type=str, default="")
        parser.add_argument('--fast_eval', type=int,  default=1)
        parser.add_argument('--eval_grid_scale', type=int,  default=1)
        parser.add_argument('--batch_norm', type=int,  default=0)
        parser.add_argument('--bn_l0', type=int,  default=0)
        parser.add_argument('--augment', type=int,  default=1)
        parser.add_argument('--rgba', type=int,  default=1)
        parser.add_argument('--symetric', type=int,  default=0)
        parser.add_argument('--num_samples', type=int,  default=0)
        parser.add_argument('--global_points', type=int,  default=1000) 
        parser.add_argument('--global_points_test', type=int,  default=1000)    
        parser.add_argument('--noise_scale', type=float,  default=[0.1])
        parser.add_argument('--categories'      , type=int,  default=[0,1,2,3,4,5,6,7,8,9,10,11,12], help='number of point samples')
        parser.add_argument('--category_names', type=int,  default=["02691156","02828884","02933112","02958343","03001627","03211117","03636649","03691459","04090263","04256520","04379243","04401088","04530566"], help='number of point samples')
        parser.add_argument('--learning_rate', type=float,  default=0.00001)
        parser.add_argument('--levelset'  , type=float,  default=0.0)
        parser.add_argument('--finetune'  , type=bool,  default=False)
        parser.add_argument('--plot_every', type=int,  default=1000)


        return parser.parse_args()




