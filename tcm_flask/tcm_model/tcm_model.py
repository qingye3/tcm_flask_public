#! /usr/bin/env python
#################################################################################
#     File Name           :     tcm_model.py
#     Created By          :     qing
#     Creation Date       :     [2018-04-20 15:05]
#     Last Modified       :     [2018-05-11 03:27]
#     Description         :      
#################################################################################
import pickle
import numpy as np


def create_mask(dim, xs, ys):
    mask1 = np.zeros(dim, dtype=bool)
    mask1[xs, :] = True
    mask2 = np.zeros(dim, dtype=bool)
    mask2[:, ys]  = True
    return mask1 * mask2

def col_to_list(col):
    ret = col.strip().split(':')
    if ret[-1] == '':
        ret = ret[:-1]
    return ret

class EMR:
    @staticmethod
    def from_serialized(serialized):
        emr = EMR()
        (emr.raw_data, emr.records, emr.herbs, emr.symptoms, emr.diseases, emr.diseases_inv, emr.symptoms_inv, emr.herbs_inv) = serialized
        return emr

    @staticmethod
    def from_raw_data_file(filename):
        with open(filename, 'rb') as fin:
            data = pickle.load(fin)
        return EMR.from_raw_data(data)

    @staticmethod
    def from_raw_data(data):
        emr = EMR()
        emr.raw_data = data

        diseases = set()
        symptoms = set()
        herbs = set()
        for record in data:
            d, s, h = [col_to_list(col) for col in record]
            diseases, symptoms, herbs = [x.union(set(y)) for x, y in zip((diseases, symptoms, herbs), (d, s, h))]
        emr.diseases, emr.symptoms, emr.herbs = list(diseases), list(symptoms), list(herbs)
        emr.diseases.append('Background')

        emr.diseases_inv, emr.symptoms_inv, emr.herbs_inv = [{x:i for i, x in enumerate(y)} for y in (emr.diseases, emr.symptoms, emr.herbs)] 
        
        records = []
        invs = emr.diseases_inv, emr.symptoms_inv, emr.herbs_inv
        for record in data:
            to_append = [[inv[x] for x in col_to_list(col)] for inv, col in zip(invs, record)]
            if not (to_append[0] and to_append[1] and to_append[2]):
                continue
            records.append(to_append)
            records[-1][0].append(emr.diseases_inv['Background'])
            records[-1] = [np.array(list(set(r)), dtype=int) for r in records[-1]]
        emr.records = records
        return emr

    def __init__(self):
        self.raw_data = None
        self.records = []
        self.herbs = []
        self.symptoms = []
        self.diseases = []

        self.diseases_inv = dict()
        self.symptoms_inv = dict()
        self.herbs_inv = dict()

    def serializable(self):
        ret = (self.raw_data,
               self.records,
               self.herbs,
               self.symptoms,
               self.diseases,
               self.diseases_inv,
               self.symptoms_inv,
               self.herbs_inv)
        return ret



class TCMModel:
    @staticmethod
    def from_saved(filename):
        with open(filename, 'br') as fin:
            serialized = pickle.load(fin)
        tcm = TCMModel()
        emr, tcm.P_d, tcm.P_t_d, tcm.P_s_d, tcm.P_h_d = serialized
        tcm.emr = EMR.from_serialized(emr)
        return tcm

    def save(self, filename):
        serialized = self.emr.serializable(), self.P_d, self.P_t_d, self.P_s_d, self.P_h_d
        with open(filename, 'bw') as fout:
            pickle.dump(serialized, fout)

    def __init__(self):
        self.emr = None
        self.P_d = None
        self.P_t_d = None
        self.P_s_d = None
        self.P_h_d = None

    def symptom_herb_corr(self):
        sh = np.sum(self.P_s_d[:, None, d] * self.P_h_d[None, :, d], axis=1)
        return sh

    def symptoms_of_disease(self, disease, top_k=10):
        d = self.emr.diseases_inv[disease]
        tops = np.argsort(self.P_s_d[:, d])[::-1][:top_k]
        return [self.emr.symptoms[s] for s in tops], self.P_s_d[tops, d]

    def herbs_of_disease(self, disease, top_k=10):
        d = self.emr.diseases_inv[disease]
        tops = np.argsort(self.P_h_d[:, d])[::-1][:top_k]
        return [self.emr.herbs[s] for s in tops], self.P_h_d[tops, d]


    def predict_disease(self, symptoms, top_k=10):
        s = np.array([self.emr.symptoms_inv[x] for x in symptoms], dtype=int)
        P_d_s = self.P_d * np.product(self.P_s_d[s, :], axis=0)
        P_d_s /= np.sum(P_d_s)
        top = np.argsort(P_d_s)[::-1][:top_k+1]
        top = [t for t in top if self.emr.diseases[t] != 'Background'][:top_k]
        return [self.emr.diseases[d] for d in top], P_d_s[top].tolist()


    def predict_herbs(self, symptoms, top_k=10):
        s = np.array([self.emr.symptoms_inv[x] for x in symptoms], dtype=int)
        
        P_d_s = self.P_d * np.product(self.P_s_d[s, :], axis=0)
        P_h_s = np.sum(self.P_h_d * P_d_s[None, :], axis=1)
        P_h_s /= np.sum(P_h_s)
        top = np.argsort(P_h_s)[::-1][:top_k]
        return [self.emr.herbs[d] for d in top], P_h_s[top].tolist()

            
    def fit_alt(self, emr_records, n_steps = 10, warm_start = False):
        self.emr = emr_records
        self.last_ll = float('-inf')
        if not warm_start:
            self.init_params_rand()
        for i in range(n_steps):
            new_ll = self.log_likelihood()
            print("{:}/{:}   ll:{:}   change:{:}".format(i, n_steps, new_ll, new_ll - self.last_ll))
            self.last_ll = new_ll
            print(self.P_d[-1])
            ss, ss_prob = self.symptoms_of_disease("便秘")
            print(ss)
            print(ss_prob)
            ss, ss_prob = self.symptoms_of_disease("慢性胃炎")
            print(ss)
            print(ss_prob)
            ss, ss_prob = self.symptoms_of_disease("反流性食管炎")
            print(ss)
            print(ss_prob)
            P_d = np.zeros(len(self.emr.diseases), dtype='float64')
            P_t_d = np.zeros((len(self.emr.records), len(self.emr.diseases)), dtype='float64')
            P_s_d = np.zeros((len(self.emr.symptoms), len(self.emr.diseases)), dtype='float64')
            P_h_d = np.zeros((len(self.emr.herbs), len(self.emr.diseases)), dtype='float64')
            for t, (d, s, h) in enumerate(self.emr.records):
                # pp = self.P_d[d][None, None, :] * self.P_t_d[t, d][None, None, :] * self.P_s_d[s, :][:, d][:, None, :] * self.P_h_d[h, :][:, d][None, :, :]
                pp = self.P_d[d] * self.P_t_d[t, d]
                pp = pp[None, :] * self.P_s_d[s, :][:, d]
                pp = pp[:, None, :] * self.P_h_d[h, :][:, d][None, :, :]
                pp /= np.sum(pp, axis=2)[:, :, None]
                P_d[d] += np.sum(pp, axis=(0, 1))
                P_t_d[t, d] += np.sum(pp, axis=(0, 1))

                SD_Mask = create_mask(P_s_d.shape, s, d)
                HD_Mask = create_mask(P_h_d.shape, h, d)
                assert(P_s_d[SD_Mask].shape == np.sum(pp, axis=1).flatten().shape)
                assert(P_h_d[HD_Mask].shape == np.sum(pp, axis=0).flatten().shape)
                P_s_d[SD_Mask] += np.sum(pp, axis=1).flatten()
                P_h_d[HD_Mask] += np.sum(pp, axis=0).flatten()

            # Update
            self.P_d = P_d
            self.P_t_d = P_t_d
            self.P_s_d = P_s_d
            self.P_h_d = P_h_d

            self.normalize_params()


    def init_params(self):
        self.P_d = np.random.random(len(self.emr.diseases), dtype='float64')
        self.P_t_d = np.zeros((len(self.emr.records), len(self.emr.diseases)), dtype='float64')

        for t, (d, s, h) in enumerate(self.emr.records):
            self.P_t_d[t, d] = 1
        self.P_s_d = np.ones((len(self.emr.symptoms), len(self.emr.diseases)), dtype='float64')
        self.P_h_d = np.ones((len(self.emr.herbs), len(self.emr.diseases)), dtype='float64')
        self.normalize_params()

    def init_params_rand(self):
        self.P_d = np.random.random(len(self.emr.diseases)).astype('float64')
        self.P_d[-1] = len(self.emr.diseases) * 0.1
        self.P_t_d = np.random.random((len(self.emr.records), len(self.emr.diseases)))

        # for t, (d, s, h) in enumerate(self.emr.records):
        #     self.P_t_d[t, d] = np.random.random((1, d.shape[0]))
        self.P_s_d = np.random.random((len(self.emr.symptoms), len(self.emr.diseases))).astype('float64')
        self.P_h_d = np.random.random((len(self.emr.herbs), len(self.emr.diseases))).astype('float64')
        self.normalize_params()

    def normalize_params(self):
        self.P_d /= np.sum(self.P_d)
        self.P_t_d /= np.sum(self.P_t_d, axis=0)[None, :]
        self.P_s_d /= np.sum(self.P_s_d, axis=0)[None, :]
        self.P_h_d /= np.sum(self.P_h_d, axis=0)[None, :]

    def log_likelihood(self):
        ret = 0
        P_d_t = self.P_d[:, None] * self.P_t_d.T
        P_d_t /= np.sum(P_d_t, axis=0)[None, :]

        for t, (d, s, h) in enumerate(self.emr.records):
            s_h_mat = self.P_s_d[s, :][:, d][:, None, :] *\
                      self.P_h_d[h, :][:, d][None, :, :] *\
                      P_d_t[d, t][None, None, :]
            ret += np.sum(np.log(np.sum(s_h_mat, axis=2)))
        return ret

        
if __name__ == '__main__':
    # emr = EMR.from_raw_data_file('../../resource/raw_records.pickle')
    # model = TCMModel()
    # model.fit_alt(emr, n_steps=1, warm_start=False)
    # model.save('../../resource/tcm_model_4.pickle')

    g = TCMModel.from_saved('../../resource/tcm_model_3.pickle')
    print(g.emr)
