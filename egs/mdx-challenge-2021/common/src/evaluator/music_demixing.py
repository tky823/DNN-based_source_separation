######################################################################################
### This is a read-only file to allow participants to run their code locally.      ###
### It will be over-writter during the evaluation, Please do not make any changes  ###
### to this file.                                                                  ###
######################################################################################

import traceback
import os
import signal
from contextlib import contextmanager
from os import listdir
from os.path import isfile, join

import soundfile as sf
import numpy as np
from evaluator import aicrowd_helpers


class TimeoutException(Exception): pass


@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Prediction timed out!")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


class MusicDemixingPredictor:
    def __init__(self):
        self.test_data_path = os.getenv("TEST_DATASET_PATH", os.getcwd() + "/data/test/")
        self.results_data_path = os.getenv("RESULTS_DATASET_PATH", os.getcwd() + "/data/results/")
        self.inference_setup_timeout = int(os.getenv("INFERENCE_SETUP_TIMEOUT_SECONDS", "900"))
        self.inference_per_music_timeout = int(os.getenv("INFERENCE_PER_MUSIC_TIMEOUT_SECONDS", "240"))
        self.partial_run = os.getenv("PARTIAL_RUN_MUSIC_NAMES", None)
        self.results = []
        self.current_music_name = None

    def get_all_music_names(self):
        valid_music_names = None
        if self.partial_run:
            valid_music_names = self.partial_run.split(',')
        music_names = []
        for folder in listdir(self.test_data_path):
            if not isfile(join(self.test_data_path, folder)):
                if valid_music_names is None or folder in valid_music_names:
                    music_names.append(folder)
        return music_names

    def get_music_folder_location(self, music_name):
        return join(self.test_data_path, music_name)

    def get_music_file_location(self, music_name, instrument=None):
        if instrument is None:
            instrument = "mixture"
            return join(self.test_data_path, music_name, instrument + ".wav")

        if not os.path.exists(self.results_data_path):
            os.makedirs(self.results_data_path)
        if not os.path.exists(join(self.results_data_path, music_name)):
            os.makedirs(join(self.results_data_path, music_name))

        return join(self.results_data_path, music_name, instrument + ".wav")

    def scoring(self):
        """
        Add scoring function in the starter kit for participant's reference
        """
        def sdr(references, estimates):
            # compute SDR for one song
            delta = 1e-7  # avoid numerical errors
            num = np.sum(np.square(references), axis=(1, 2))
            den = np.sum(np.square(references - estimates), axis=(1, 2))
            num += delta
            den += delta
            return 10 * np.log10(num  / den)

        music_names = self.get_all_music_names()
        instruments = ["bass", "drums", "other", "vocals"]
        scores = {}
        for music_name in music_names:
            print("Evaluating for: %s" % music_name)
            scores[music_name] = {}
            references = []
            estimates = []
            for instrument in instruments:
                reference_file = join(self.test_data_path, music_name, instrument + ".wav")
                estimate_file = self.get_music_file_location(music_name, instrument)
                reference, _ = sf.read(reference_file)
                estimate, _ = sf.read(estimate_file)
                references.append(reference)
                estimates.append(estimate)
            references = np.stack(references)
            estimates = np.stack(estimates)
            references = references.astype(np.float32)
            estimates = estimates.astype(np.float32)
            song_score = sdr(references, estimates).tolist()
            scores[music_name]["sdr_bass"] = song_score[0]
            scores[music_name]["sdr_drums"] = song_score[1]
            scores[music_name]["sdr_other"] = song_score[2]
            scores[music_name]["sdr_vocals"] = song_score[3]
            scores[music_name]["sdr"] = np.mean(song_score)
        return scores


    def evaluation(self):
        """
        Admin function: Runs the whole evaluation
        """
        aicrowd_helpers.execution_start()
        try:
            with time_limit(self.inference_setup_timeout):
                self.prediction_setup()
        except NotImplementedError:
            print("prediction_setup doesn't exist for this run, skipping...")

        aicrowd_helpers.execution_running()

        music_names = self.get_all_music_names()

        for music_name in music_names:
            with time_limit(self.inference_per_music_timeout):
                self.prediction(mixture_file_path=self.get_music_file_location(music_name),
                                bass_file_path=self.get_music_file_location(music_name, "bass"),
                                drums_file_path=self.get_music_file_location(music_name, "drums"),
                                other_file_path=self.get_music_file_location(music_name, "other"),
                                vocals_file_path=self.get_music_file_location(music_name, "vocals"),
                )
                
            if not self.verify_results(music_name):
                raise Exception("verification failed, demixed files not found.")
        aicrowd_helpers.execution_success()

    def run(self):
        try:
            self.evaluation()
        except Exception as e:
            error = traceback.format_exc()
            print(error)
            aicrowd_helpers.execution_error(error)
            if not aicrowd_helpers.is_grading():
                raise e

    def prediction_setup(self):
        """
        You can do any preprocessing required for your codebase here : 
            like loading your models into memory, etc.
        """
        raise NotImplementedError

    def prediction(self, music_name, mixture_file_path, bass_file_path, drums_file_path, other_file_path,
                   vocals_file_path):
        """
        This function will be called for all the flight during the evaluation.
        NOTE: In case you want to load your model, please do so in `inference_setup` function.
        """
        raise NotImplementedError

    def verify_results(self, music_name):
        """
        This function will be called to check all the files exist and other verification needed.
        (like length of the wav files)
        """
        valid = True
        valid = valid and os.path.isfile(self.get_music_file_location(music_name, "vocals"))
        valid = valid and os.path.isfile(self.get_music_file_location(music_name, "bass"))
        valid = valid and os.path.isfile(self.get_music_file_location(music_name, "drums"))
        valid = valid and os.path.isfile(self.get_music_file_location(music_name, "other"))
        return valid
