import yaml


class Model_config:
    def __init__(self, yaml_path=None):
        self.model_yml_path = yaml_path
        assert self.model_yml_path is not None
        model_file = open(self.model_yml_path, 'r')
        model_dict = yaml.load(model_file, Loader=yaml.FullLoader)
        model_file.close()

        self.GPU_ID = model_dict['GENERAL']['GPU_ID']
        self.SEED = model_dict['GENERAL']['SEED']

        self.MODE = model_dict['DATA']['MODE']
        self.TRAIN_DATASET = model_dict['DATA']['TRAIN_DATASET']
        self.TEST_DATASET = model_dict['DATA']['TEST_DATASET']
        self.HEIGHT = model_dict['DATA']['HEIGHT']
        self.WIDTH = model_dict['DATA']['WIDTH']
        self.frames_each_video = model_dict['DATA']['frames_each_video']

        self.N_G = model_dict['TRAIN']['N_G']
        self.N_D = model_dict['TRAIN']['N_D']
        self.LOADER = model_dict['TRAIN']['LOADER']
        self.MODEL_DIR = model_dict['TRAIN']['MODEL_DIR']
        self.EPOCHS = model_dict['TRAIN']['EPOCHS']
        self.BASE_LR = model_dict['TRAIN']['BASE_LR']
        self.BATCH_SIZE = model_dict['TRAIN']['BATCH_SIZE']
        self.CROP_SIZE = model_dict['TRAIN']['CROP_SIZE']
        self.RESIZE_H = model_dict['TRAIN']['RESIZE_H']
        self.RESIZE_W = model_dict['TRAIN']['RESIZE_W']
        self.SAVE_ITER = model_dict['TRAIN']['SAVE_ITER']
        self.LOAD_EPOCH = model_dict['TRAIN']['LOAD_EPOCH']
        self.LOGS_DIR = model_dict['TRAIN']['LOGS_DIR']
        self.VISUALS_DIR = model_dict['TRAIN']['VISUALS_DIR']
        self.NETS_DIR = model_dict['TRAIN']['NETS_DIR']
        self.DECAY_EPOCH = model_dict['TRAIN']['DECAY_EPOCH']
        self.DECAY_FACTOR = model_dict['TRAIN']['DECAY_FACTOR']
        self.MIN_LR = model_dict['TRAIN']['MIN_LR']
        self.WEIGHT_PEC = model_dict['TRAIN']['WEIGHT_PEC']
        self.WEIGHT_L1 = model_dict['TRAIN']['WEIGHT_L1']
        self.NUM_AUX_FRAMES = model_dict['TRAIN']['NUM_AUX_FRAMES']
        self.FRAME_INTERVAL = model_dict['TRAIN']['FRAME_INTERVAL']
        self.num_res_blocks = model_dict['TRAIN']['num_res_blocks']
        self.n_feats = model_dict['TRAIN']['n_feats']
        self.res_scale = model_dict['TRAIN']['res_scale']
        self.backbone = model_dict['TRAIN']['backbone']

        #  use_flow: False  # use estimated optical flow?
        self.use_occu = model_dict['TRAIN']['use_occu']
        self.use_shuffle = model_dict['TRAIN']['use_shuffle']
        self.use_temporal = model_dict['TRAIN']['use_temporal']
        self.weight_t = model_dict['TRAIN']['weight_t']
        self.temporal_begin_epoch = model_dict['TRAIN']['temporal_begin_epoch']
        self.temporal_loss_mode = model_dict['TRAIN']['temporal_loss_mode']
        self.k_sizes = model_dict['TRAIN']['k_sizes']
        #self.k_weights = model_dict['TRAIN']['k_weights']

        self.VAL_RESULT_DIR = model_dict['TEST']['VAL_RESULT_DIR']
        self.VAL_TIME = model_dict['TEST']['VAL_TIME']
        self.TEST_EPOCH = model_dict['TEST']['TEST_EPOCH']
        self.TEST_RESULT_DIR = model_dict['TEST']['TEST_RESULT_DIR']
        self.SAVE_IMG = model_dict['TEST']['SAVE_IMG']
        self.have_gt = model_dict['TEST']['have_gt']

        self.WARM_UP_ITER = model_dict['SOLVER']['WARM_UP_ITER']
        self.WARM_UP_FACTOR = model_dict['SOLVER']['WARM_UP_FACTOR']
        self.MAX_ITER = model_dict['SOLVER']['MAX_ITER']
        self.T_PERIOD = model_dict['SOLVER']['T_PERIOD']



if __name__ == '__main__':
    model_path = "D:\labs\Video_Consistency\VideoDemoireing_Model\config\config.yaml"
    class_obj = Model_config(model_path)
