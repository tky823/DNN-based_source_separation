import os
import shutil
import subprocess
import uuid

from mir_eval.separation import bss_eval_sources
import torch
import torchaudio

from utils.bss import bss_eval_sources

BITS_PER_SAMPLE_WSJ0 = 16
MIN_PESQ = -0.5

class AdhocTester:
    def __init__(self, method, loader, criterion, args):
        self.loader = loader

        self.method = method

        self.criterion = criterion

        self._reset(args)

    def _reset(self, args):
        self.sample_rate = args.sample_rate
        self.n_sources = args.n_sources

        self.out_dir = args.out_dir

        if self.out_dir is not None:
            self.out_dir = os.path.abspath(args.out_dir)
            os.makedirs(self.out_dir, exist_ok=True)

    def run(self):
        test_loss = 0
        test_loss_improvement = 0
        test_sdr_improvement = 0
        test_sir_improvement = 0
        test_sar = 0
        test_pesq = 0
        n_pesq_error = 0
        n_test = len(self.loader.dataset)

        print("ID, Loss, Loss improvement, SDR improvement, SIR improvement, SAR, PESQ", flush=True)

        tmp_ID = str(uuid.uuid4())
        tmp_dir = os.path.join(os.getcwd(), 'tmp-{}'.format(tmp_ID))
        os.makedirs(tmp_dir, exist_ok=True)
        shutil.copy('./PESQ', os.path.join(tmp_dir, 'PESQ'))
        os.chdir(tmp_dir)

        with torch.no_grad():
            for idx, (mixture, sources, segment_IDs) in enumerate(self.loader):
                loss_mixture = self.criterion(mixture, sources, batch_mean=False)
                loss_mixture = loss_mixture.sum(dim=0)

                output = self.method(mixture, sources)
                loss = self.criterion(output, sources, batch_mean=False)
                loss = loss.sum(dim=0)
                loss_improvement = loss_mixture.item() - loss.item()

                mixture = mixture[0].squeeze(dim=0) # -> (T,)
                sources = sources[0] # -> (n_sources, T)
                oracle_sources = output[0] # -> (n_sources, T)
                segment_IDs = segment_IDs[0] # -> <str>

                repeated_mixture = torch.tile(mixture, (self.n_sources, 1))
                result_oracle = bss_eval_sources(
                    reference_sources=sources,
                    estimated_sources=oracle_sources
                )
                result_mixed = bss_eval_sources(
                    reference_sources=sources,
                    estimated_sources=repeated_mixture
                )

                sdr_improvement = torch.mean(result_oracle[0] - result_mixed[0])
                sir_improvement = torch.mean(result_oracle[1] - result_mixed[1])
                sar = torch.mean(result_oracle[2])

                mixture_ID = segment_IDs

                # Generate random number temporary wav file.
                random_ID = str(uuid.uuid4())

                if idx < 10 and self.out_dir is not None:
                    mixture_path = os.path.join(self.out_dir, "{}.wav".format(mixture_ID))
                    signal = mixture.unsqueeze(dim=0) if mixture.dim() == 1 else mixture
                    torchaudio.save(mixture_path, signal, sample_rate=self.sample_rate, bits_per_sample=BITS_PER_SAMPLE_WSJ0)

                for order_idx in range(self.n_sources):
                    source, oracle_source = sources[order_idx], oracle_sources[order_idx]
                    # Target
                    if idx < 10 and  self.out_dir is not None:
                        source_path = os.path.join(self.out_dir, "{}_{}-target.wav".format(mixture_ID, order_idx + 1))
                        signal = source.unsqueeze(dim=0) if source.dim() == 1 else source
                        torchaudio.save(source_path, signal, sample_rate=self.sample_rate, bits_per_sample=BITS_PER_SAMPLE_WSJ0)
                    source_path = "tmp-{}-target_{}.wav".format(order_idx + 1, random_ID)
                    signal = source.unsqueeze(dim=0) if source.dim() == 1 else source
                    torchaudio.save(source_path, signal, sample_rate=self.sample_rate, bits_per_sample=BITS_PER_SAMPLE_WSJ0)

                    # Oracle source
                    if idx < 10 and  self.out_dir is not None:
                        oracle_path = os.path.join(self.out_dir, "{}_{}-oracle.wav".format(mixture_ID, order_idx + 1))
                        signal = oracle_source.unsqueeze(dim=0) if oracle_source.dim() == 1 else oracle_source
                        torchaudio.save(oracle_path, signal, sample_rate=self.sample_rate, bits_per_sample=BITS_PER_SAMPLE_WSJ0)
                    oracle_path = "tmp-{}-oracle_{}.wav".format(order_idx + 1, random_ID)
                    signal = oracle_source.unsqueeze(dim=0) if oracle_source.dim() == 1 else oracle_source
                    torchaudio.save(oracle_path, signal, sample_rate=self.sample_rate, bits_per_sample=BITS_PER_SAMPLE_WSJ0)

                pesq = 0

                for source_idx in range(self.n_sources):
                    source_path = "tmp-{}-target_{}.wav".format(source_idx + 1, random_ID)
                    oracle_path = "tmp-{}-oracle_{}.wav".format(source_idx + 1, random_ID)

                    command = "./PESQ +{} {} {}".format(self.sample_rate, source_path, oracle_path)
                    command += " | grep Prediction | awk '{print $5}'"
                    pesq_output = subprocess.check_output(command, shell=True)
                    pesq_output = pesq_output.decode().strip()

                    if pesq_output == '':
                        # If processing error occurs in PESQ software, it is regarded as PESQ score is -0.5. (minimum of PESQ)
                        n_pesq_error += 1
                        pesq += MIN_PESQ
                    else:
                        pesq += float(pesq_output)

                    subprocess.call("rm {}".format(source_path), shell=True)
                    subprocess.call("rm {}".format(oracle_path), shell=True)

                pesq /= self.n_sources
                print("{}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}".format(mixture_ID, loss.item(), loss_improvement, sdr_improvement.item(), sir_improvement.item(), sar.item(), pesq), flush=True)

                test_loss += loss.item()
                test_loss_improvement += loss_improvement
                test_sdr_improvement += sdr_improvement.item()
                test_sir_improvement += sir_improvement.item()
                test_sar += sar.item()
                test_pesq += pesq

        os.chdir("../") # back to the original directory
        shutil.rmtree(tmp_dir)

        test_loss /= n_test
        test_loss_improvement /= n_test
        test_sdr_improvement /= n_test
        test_sir_improvement /= n_test
        test_sar /= n_test
        test_pesq /= n_test

        print("Loss: {:.3f}, loss improvement: {:3f}, SDR improvement: {:3f}, SIR improvement: {:3f}, SAR: {:3f}, PESQ: {:.3f}".format(test_loss, test_loss_improvement, test_sdr_improvement, test_sir_improvement, test_sar, test_pesq))
        print("Evaluation of PESQ returns error {} times.".format(n_pesq_error))
