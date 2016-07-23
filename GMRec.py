#!usr/bin/env python
#coding:utf-8

from util import logger
from data.DataModel import DataModel
from RecommenderContext import RecommenderContext

from util.readconf import ReadConfig
from cf_rating import PMF, BPMF, BPTF


class GMRec:
    def __init__(self, config_file, algorithm_name):
        self.config_file = config_file
        self.algorithm_name = algorithm_name
        self.all_algorithms = {
            'PMF': PMF, 'BPMF': BPMF, 'BPTF': BPTF
        }

    def _set_logger(self, config_handler):
        result_file = config_handler.get_parameter_string("Output", "logger") + "{0}_Result.log".format(self.algorithm_name)
        process_file = config_handler.get_parameter_string("Output", "logger") + "{0}_Process.log".format(self.algorithm_name)
        logger1 = {'Result': logger.Result(result_file), 'Process': logger.Process(process_file)}
        return logger1

    def run(self):
        config_handler = ReadConfig(self.config_file)
        loggerc = self._set_logger(config_handler)
        data_model = DataModel(config_handler)
        recommender_context = RecommenderContext(config_handler, data_model, loggerc)

        recommender_context.get_logger()['Process'].debug("\n" + "#"*50 + "Start" + '#'*50)
        recommender_context.get_logger()['Result'].debug("\n" + "#"*50 + "Start" + '#'*50)

        recommender_context.get_logger()['Process'].debug("Build data model")
        recommender_context.get_data_model().build_data_model()

        experiment_num = recommender_context.get_config().get_parameter_int("splitter", "experiment_num")
        for experiment_id in range(experiment_num):
            recommender_context.get_logger()['Process'].debug("The {0}th experiment.".format(experiment_id))
            recommender_context.get_logger()['Result'].debug("The {0}th experiment.".format(experiment_id))

            recommender_context.get_logger()['Process'].debug("Split dataset into train and test")
            save_path = recommender_context.get_config().get_parameter_string("splitter", "save_path")
            recommender_context.experiment_id = experiment_id
            recommender_context.get_data_model().get_data_splitter().split_data(save_path, experiment_id)

            recommender_context.get_logger()['Process'].debug("Enter into training ....")
            algorithm = self.all_algorithms[self.algorithm_name](recommender_context)
            algorithm.run()

        recommender_context.get_logger()['Process'].debug("\n" + "#"*50 + "Finish" + "#"*50)