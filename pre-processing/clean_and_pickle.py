#! /usr/bin/env python
#################################################################################
#     File Name           :     clean_and_pickle.py
#     Created By          :     qing
#     Creation Date       :     [2018-04-20 14:09]
#     Last Modified       :     [2018-04-20 14:26]
#     Description         :      
#################################################################################
import csv
import pickle
if __name__ == '__main__':
    with open('../resource/HIS_tuple_word.txt') as fin:
        data = []
        for line in csv.reader(fin, delimiter='\t'):
            if len(line) != 6:
                continue
            data.append([line[0], line[4], line[5]])
    with open('../resource/raw_records.pickle', 'bw') as fout:
        pickle.dump(data, fout)
