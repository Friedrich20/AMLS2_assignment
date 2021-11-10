#!/usr/local/bin/python
# -*- coding:utf-8 -*-

import gc
import logging
import os
import time

from A.A_sentiment_classification import A
from B.B_topic_based_sentiment_classification import B

# # ======================================================================================================================
# # Constants and paths
home_dir = os.path.abspath(os.curdir)  # the current directory
data_dir = os.path.join(home_dir, 'Codes', 'Datasets')
data_path_A = os.path.join(data_dir, '4A-English',
                           'SemEval2017-task4-dev.subtask-A.english.INPUT.txt')
data_path_B = os.path.join(data_dir, '4B-English',
                           'SemEval2017-task4-dev.subtask-BD.english.INPUT.txt')
log_path = os.path.join(home_dir, 'Codes', 'helper',
                        'base_log.log')  # the path of log file

# # ======================================================================================================================
# # The configuration of logging module (feel free to change the logging level)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename=log_path,
                    filemode='w')

# # ======================================================================================================================
# # Main program
logging.info('*****************************************')
logging.info('**********[Main program starts]**********')
logging.info('*****************************************\n')

start_time = time.time()

# # ======================================================================================================================
# # Task A
logging.info('**********[Task A starts]**********')

task_a = A(data_path_A)
X_train, X_test, y_train, y_test = task_a.preprocess_data()
acc_A_train, acc_A_test = task_a.train_model_lstm(
    X_train, X_test, y_train, y_test, loaded_model=False)  # Train a new model by default, set to True to use the existing trained model

##### for debugging only #####
logging.debug(f'TA:{acc_A_train},{acc_A_test}')

gc.collect()

logging.info('**********[Task A ends]**********')
logging.info('***********************************')
# # ======================================================================================================================
# # Task B
logging.info('**********[Task B starts]**********')

task_b = B(data_path_B)
X_train, X_test, y_train, y_test = task_b.preprocess_data()
acc_B_train, acc_B_test = task_b.train_model_lstm(
    X_train, X_test, y_train, y_test, loaded_model=False)  # Train a new model by default, set to True to use the existing trained model

##### for debugging only #####
logging.debug(f'TB:{acc_B_train},{acc_B_test}')

gc.collect()

logging.info('**********[Task B ends]**********')
logging.info('***********************************')
# # ======================================================================================================================
# Print out your results with following format:
print('TA:{},{};TB:{},{};'.format(
    acc_A_train, acc_A_test, acc_B_train, acc_B_test))

end_time = time.time()
elapsed_time = end_time - start_time

logging.info(
    f'[Total execution time]: {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}')

logging.info('*****************************************')
logging.info('***********[Main program ends]***********')
logging.info('*****************************************\n')
