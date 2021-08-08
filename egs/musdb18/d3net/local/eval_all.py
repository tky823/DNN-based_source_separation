#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

from utils.utils import set_seed
from driver import EvaluaterBase as Evaluater

parser = argparse.ArgumentParser(description="Evaluation of D3Net")

parser.add_argument('--musdb18_root', type=str, default=None, help='Path to MUSDB18')
parser.add_argument('--estimated_musdb18_root', type=str, default=None, help='Path to estimated MUSDB18')
parser.add_argument('--json_dir', type=str, default=None, help='Output json directory')
parser.add_argument('--seed', type=int, default=42, help='Random seed')

def main(args):
    set_seed(args.seed)
    
    evaluater = Evaluater(args)
    evaluater.run()
    
if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    main(args)
