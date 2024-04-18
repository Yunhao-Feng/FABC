"""导包"""
import math
import os
import logging


logger = logging.getLogger("logger")


class Helper:
    def __init__(self, current_time, params, name):
        self.current_time = current_time
        self.target_model = None
        self.local_model = None
        
        self.train_data = None
        self.test_data = None
        self.poisoned_data = None
        self.test_data_poison = None
        
        self.params = params
        self.name = name
        self.best_loss = math.inf
        self.folder_path = f'{os.getcwd()}/saved_models/model_{self.name}_{current_time}'
        try:
            os.mkdir(self.folder_path)  # 创建储存文件夹
        except FileExistsError:
            logger.info("文件已经存在！")  # 输出异常
        logger.addHandler(logging.FileHandler(filename=f'{self.folder_path}/log.txt'))
        logger.addHandler(logging.StreamHandler())
        logger.setLevel(logging.DEBUG)
        logger.info(f'current path: {self.folder_path}')
        
        if not self.params.get('environment_name', False):
            self.params['environment_name'] = self.name
        self.params['current_time'] = self.current_time
        self.params['folder_path'] = self.folder_path
        self.fg= FoolsGold(use_memory=self.params['fg_use_memory'])


class FoolsGold(object):
    def __init__(self, use_memory=False):
        self.memory = None
        self.memory_dict = dict()
        self.wv_history = []
        self.use_memory = use_memory
        