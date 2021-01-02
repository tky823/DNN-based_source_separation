#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import json
import glob

sources = ['vocals', 'bass', 'drums', 'others']

parser = argparse.ArgumentParser("Prepare json file of DSD100 for audio source separation")

parser.add_argument('--source_dir', type=str, default=None, help='Path for source directory')
parser.add_argument('--mixture_dir', type=str, default=None, help='Path for mixture directory')
parser.add_argument('--json_path', type=str, default=None, help='Path for json file')

def main(args):
    json_data = make_json_data(args.source_dir, args.mixture_dir, args.json_path)
        
    with open(args.json_path, 'w') as f:
        json.dump(json_data, f, indent=4)

def make_json_data(source_dir, mixture_dir, json_path):
    json_dir = os.path.dirname(json_path)
    folder_name = os.path.basename(json_dir)
    os.makedirs(json_dir, exist_ok=True)

    json_data = []

    titles = sorted(glob.glob(os.path.join(source_dir, "*")))
    titles = [os.path.basename(title) for title in titles]

    for title in titles:
        data = {
            'title': title,
            'sources': {},
            'mixture': {}
        }

        for source in sources:
            data['sources'][source] = {}
            data['sources'][source]['path'] = os.path.join(source_dir, title, '{}.wav'.format(source))
        
        data['mixture']['path'] = os.path.join(mixture_dir, title, 'mixture.wav')
        json_data.append(data)

    return json_data

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    main(args)