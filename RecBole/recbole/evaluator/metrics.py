# -*- encoding: utf-8 -*-
# @Time    :   2020/08/04
# @Author  :   Kaiyuan Li
# @email   :   tsotfsk@outlook.com

# UPDATE
# @Time    :   2020/08/12, 2021/8/29, 2020/9/16, 2021/7/2
# @Author  :   Kaiyuan Li, Zhichao Feng, Xingyu Pan, Zihan Lin
# @email   :   tsotfsk@outlook.com, fzcbupt@gmail.com, panxy@ruc.edu.cn, zhlin@ruc.edu.cn

r"""
recbole.evaluator.metrics
############################

Suppose there is a set of :math:`n` items to be ranked. Given a user :math:`u` in the user set :math:`U`,
we use :math:`\hat R(u)` to represent a ranked list of items that a model produces, and :math:`R(u)` to
represent a ground-truth set of items that user :math:`u` has interacted with. For top-k recommendation, only
top-ranked items are important to consider. Therefore, in top-k evaluation scenarios, we truncate the
recommendation list with a length :math:`K`. Besides, in loss-based metrics, :math:`S` represents the
set of user(u)-item(i) pairs, :math:`\hat r_{u i}` represents the score predicted by the model,
:math:`{r}_{u i}` represents the ground-truth labels.

"""

from logging import getLogger

import numpy as np

from recbole.evaluator.base_metric import AbstractMetric, TopkMetric
from recbole.utils import EvaluatorType

from collections import Counter

import torch
from scipy.spatial.distance import pdist #for VoCD
import numpy.ma as ma
from pytest import approx
# TopK Metrics

class NDCG(TopkMetric):
    r"""NDCG_ (also known as normalized discounted cumulative gain) is a measure of ranking quality,
    where positions are discounted logarithmically. It accounts for the position of the hit by assigning
    higher scores to hits at top ranks.

    .. _NDCG: https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Normalized_DCG

    .. math::
        \mathrm {NDCG@K} = \frac{1}{|U|}\sum_{u \in U} (\frac{1}{\sum_{i=1}^{\min (|R(u)|, K)}
        \frac{1}{\log _{2}(i+1)}} \sum_{i=1}^{K} \delta(i \in R(u)) \frac{1}{\log _{2}(i+1)})

    :math:`\delta(·)` is an indicator function.
    """

    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, dataobject):
        pos_index, pos_len = self.used_info(dataobject)
        result = self.metric_info(pos_index, pos_len)
        metric_dict = self.topk_result('ndcg', result)
        return metric_dict

    def metric_info(self, pos_index, pos_len):
        len_rank = np.full_like(pos_len, pos_index.shape[1])
        idcg_len = np.where(pos_len > len_rank, len_rank, pos_len)

        iranks = np.zeros_like(pos_index, dtype=np.float)
        iranks[:, :] = np.arange(1, pos_index.shape[1] + 1)
        idcg = np.cumsum(1.0 / np.log2(iranks + 1), axis=1)
        for row, idx in enumerate(idcg_len):
            idcg[row, idx:] = idcg[row, idx - 1]

        ranks = np.zeros_like(pos_index, dtype=np.float)
        ranks[:, :] = np.arange(1, pos_index.shape[1] + 1)
        dcg = 1.0 / np.log2(ranks + 1)
        dcg = np.cumsum(np.where(pos_index, dcg, 0), axis=1)

        result = dcg / idcg
        return result


class RelMetrics(TopkMetric):
    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, dataobject):
        pos_index, pos_len = self.used_info(dataobject)

        hit, result = self.hit(pos_index)
        metric_dict = self.topk_result('HR', hit)

        mrr = self.mrr(pos_index)
        metric_dict.update(self.topk_result('MRR', mrr))

        prec = self.prec(pos_index)
        metric_dict.update(self.topk_result('P', prec))

        MAP = self.MAP(pos_index, pos_len)
        metric_dict.update(self.topk_result('MAP', MAP))

        recall = self.recall(result, pos_len)
        metric_dict.update(self.topk_result('R', recall))

        NDCG = self.ndcg(pos_index, pos_len)
        metric_dict.update(self.topk_result('NDCG', NDCG))
     
        return metric_dict

    def prec(self, pos_index):
        return pos_index.cumsum(axis=1) / np.arange(1, pos_index.shape[1] + 1)

    def ndcg(self, pos_index, pos_len):
        len_rank = np.full_like(pos_len, pos_index.shape[1])
        idcg_len = np.where(pos_len > len_rank, len_rank, pos_len)

        iranks = np.zeros_like(pos_index, dtype=np.float)
        iranks[:, :] = np.arange(1, pos_index.shape[1] + 1)
        idcg = np.cumsum(1.0 / np.log2(iranks + 1), axis=1)
        for row, idx in enumerate(idcg_len):
            idcg[row, idx:] = idcg[row, idx - 1]

        ranks = np.zeros_like(pos_index, dtype=np.float)
        ranks[:, :] = np.arange(1, pos_index.shape[1] + 1)
        dcg = 1.0 / np.log2(ranks + 1)
        dcg = np.cumsum(np.where(pos_index, dcg, 0), axis=1)

        result = dcg / idcg
        return result

    def recall(self, result, pos_len):
        return result / pos_len.reshape(-1, 1)

    def MAP(self, pos_index, pos_len):
        pre = pos_index.cumsum(axis=1) / np.arange(1, pos_index.shape[1] + 1)
        sum_pre = np.cumsum(pre * pos_index.astype(np.float), axis=1)
        len_rank = np.full_like(pos_len, pos_index.shape[1])
        actual_len = np.where(pos_len > len_rank, len_rank, pos_len)
        result = np.zeros_like(pos_index, dtype=np.float)
        for row, lens in enumerate(actual_len):
            ranges = np.arange(1, pos_index.shape[1] + 1)
            ranges[lens:] = ranges[lens - 1]
            result[row] = sum_pre[row] / ranges
        return result

    def hit(self, pos_index):
        result = np.cumsum(pos_index, axis=1)
        return (result > 0).astype(int), result

    def mrr(self, pos_index):
        idxs = pos_index.argmax(axis=1)
        result = np.zeros_like(pos_index, dtype=np.float)
        for row, idx in enumerate(idxs):
            if pos_index[row, idx] > 0:
                result[row, idx:] = 1 / (idx + 1)
            else:
                result[row, idx:] = 0
        return result

class FairWORel(AbstractMetric):
    metric_type = EvaluatorType.RANKING
    smaller = False
    metric_need = ['rec.items', 'data.num_items',"rec.score"]

    def __init__(self, config):
        super().__init__(config)
        self.topk = config['topk']

    def used_info(self, dataobject):
        """Get the matrix of recommendation items and number of items in total item set"""
        item_matrix = dataobject.get('rec.items')
        num_items = dataobject.get('data.num_items') - 1
        return item_matrix.numpy(), num_items

    def calculate_metric(self, dataobject):
        item_matrix, num_items = self.used_info(dataobject)

        metric_dict = {}
        for k in self.topk:
            item_matrix_k = item_matrix[:, :k]
            item_count = np.unique(item_matrix_k, return_counts=True)[1]
            slot = item_matrix_k.size

            floor_km_n = slot//num_items
            km_mod_n = slot%num_items

            jain_ori, jain_our = self.get_jain(item_count, slot, num_items, k, floor_km_n, km_mod_n)
            qf_ori, qf_our = self.get_QF(item_matrix_k, slot, num_items, k)
            ent_ori, ent_our = self.get_entropy(item_count,slot,num_items, k, floor_km_n, km_mod_n)
            gini_ori, gini_our, gini_w_ori, gini_w_our = self.get_gini(item_matrix_k,item_count,slot,num_items, k, km_mod_n)
            vocd_ori = self.get_vocd(item_count,slot, k)
            fsat_ori, fsat_our = self.get_fsat(item_count,slot,num_items, k)

            key = '{}@{}'.format('Jain_ori', k)
            metric_dict[key] = round(jain_ori, self.decimal_place)

            key = '{}@{}'.format('Jain_our', k)
            metric_dict[key] = round(jain_our, self.decimal_place)

            key = '{}@{}'.format('QF_ori', k)
            metric_dict[key] = round(qf_ori, self.decimal_place)
        
            key = '{}@{}'.format('QF_our', k)
            metric_dict[key] = round(qf_our, self.decimal_place)

            key = '{}@{}'.format('Ent_ori', k)
            metric_dict[key] = round(ent_ori, self.decimal_place)

            key = '{}@{}'.format('Ent_our', k)
            metric_dict[key] = round(ent_our, self.decimal_place)

            key = '{}@{}'.format('Gini_ori', k)
            metric_dict[key] = round(gini_ori, self.decimal_place)

            key = '{}@{}'.format('Gini_our', k)
            metric_dict[key] = round(gini_our, self.decimal_place)

            key = '{}@{}'.format('Gini-w_ori', k)
            metric_dict[key] = round(gini_w_ori, self.decimal_place)
    
            key = '{}@{}'.format('Gini-w_our', k)
            metric_dict[key] = round(gini_w_our, self.decimal_place)

            key = '{}@{}'.format('FSat_ori', k)
            metric_dict[key] = round(fsat_ori, self.decimal_place)
        
            key = '{}@{}'.format('FSat_our', k)
            metric_dict[key] = round(fsat_our, self.decimal_place)

            key = '{}@{}'.format('VoCD_ori', k)
            metric_dict[key] = round(vocd_ori, self.decimal_place)
        
        return metric_dict

    def get_vocd(self,item_count, slot, k):
        item_count = item_count.reshape(-1,1)
        sum_result = pdist(item_count, lambda u, v: (np.abs((u-v)/max(u,v))).sum()) #does not include comparison with self
        vocd_ori = sum_result.mean()

        return vocd_ori

    def get_fsat(self, item_count, slot, num_items, k):
        n = num_items
        maximinshare = slot//n

        sat_items = (item_count >= maximinshare).sum()
        if maximinshare == 0:
            sat_items += n - item_count.shape[0]
        
        fsat_ori = sat_items / n

        fsat_our = (fsat_ori -  k/n)/ (1- k/n)

        return fsat_ori, fsat_our

    def get_gini(self,item_matrix,item_count,slot,num_items, k,km_mod_n):
        #gini_ori
        num_recommended_items = item_count.size
        item_count.sort()
        sorted_count = item_count

        idx = np.arange(num_items - num_recommended_items + 1, num_items + 1)
        gini_index = np.sum((2 * idx - num_items - 1) * sorted_count) / slot
        gini_index /= num_items

        #gini_w_ori
        unique_items = np.unique(item_matrix)
        weights = [(1/np.log2(np.where(item_matrix==item)[1]+2)).sum() for item in unique_items]
        total_weights = sum(weights)

        sorted_count = np.array(sorted(weights),dtype='float64')
        num_recommended_items = sorted_count.shape[0]

        idx = np.arange(num_items - num_recommended_items + 1, num_items + 1)
        gini_w_ori = np.sum((2 * idx - num_items - 1) * sorted_count) / total_weights
        gini_w_ori /= num_items

        #gini_our
        n = num_items
        gini_min = (n-km_mod_n)*km_mod_n/(n*slot)
        numerator = gini_index - gini_min
        denom = 1 - k/n - gini_min
        gini_our = numerator/denom

        rounded_gini_our = round(gini_our,15)
        assert approx(gini_our) == approx(rounded_gini_our)
        gini_our = rounded_gini_our

        if approx(gini_our) == 0:
            gini_our = 0
        elif approx(gini_our) == 1:
            gini_our = 1
        else:
            assert gini_our >= 0 and gini_our <=1, "need to be non-negative and not more than 1"

        assert abs(gini_our) == gini_our

        #gini_w_our
        m = slot/k      
        log_array = np.flip(1/np.log2(np.arange(k)+2))

        denom = n * m * log_array.sum()

        ell = np.arange(k)+(n-k+1)

        count = 0
        m=int(m)
        for l in range(1,k+1):
                for j in range(n-l*m+1, n-l*m+m+1):
                        count += (2*j-n-1) * 1/np.log2(l+1)
        numerator_gini_w_min = count

        numerator_gini_w_max = ((2*ell-n-1) * m *log_array).sum() #the m will cancel out with the denom
        
        if slot <= num_items:
            gini_w_min = numerator_gini_w_min/denom
        else:
            gini_w_min = 0
        gini_w_max = numerator_gini_w_max/denom

        gini_w_our = (gini_w_ori-gini_w_min) / (gini_w_max-gini_w_min)
        return gini_index, abs(gini_our), gini_w_ori, gini_w_our

    def get_entropy(self, item_count,slot,num_items, k, floor_km_n, km_mod_n):

        p = item_count/slot
        p_log_p = -p * (np.log(p)/np.log(num_items))
        ent_our_before_norm = p_log_p.sum()

        log_n_k = (np.log(k)/np.log(num_items))
        numerator = ent_our_before_norm - log_n_k

        if slot >= num_items:

            x = floor_km_n/slot 
            ent_max = -(num_items - km_mod_n) * (x*np.log(x)/np.log(num_items))
            
            x = (floor_km_n + 1)/slot
            ent_max -= km_mod_n * (x*np.log(x)/np.log(num_items))
            denom = ent_max - log_n_k

        else:
            denom = (np.log(slot/k)/np.log(num_items))

        ent_our = numerator/denom

        item_count = np.append(item_count, np.zeros(num_items-item_count.size)) #include count of unexposed items

        p = item_count/slot
        p_log_p = -p * (np.log(p)/np.log(num_items))
        ent_ori = p_log_p.sum()


        if not np.isnan(ent_ori):
            rounded_ent_ori = round(ent_ori,15)
            assert approx(ent_ori) == approx(rounded_ent_ori)
            ent_ori = rounded_ent_ori
            assert ent_ori >= 0 and ent_ori <=1, "need to be non-negative and not more than 1"

        rounded_ent_our = round(ent_our,15)
        assert approx(ent_our) == approx(rounded_ent_our)
        ent_our = rounded_ent_our

        if approx(ent_our) == 0:
            ent_our = 0
        elif approx(ent_our) == 1:
            ent_our = 1
        else:
            assert ent_our >= 0 and ent_our <=1, "need to be non-negative and not more than 1"

        return ent_ori, ent_our

    def get_jain(self, item_count, slot, num_items, k, floor_km_n, km_mod_n):

        numerator = slot**2
        sum_of_squared_count = (item_count**2).sum()

        assert sum_of_squared_count >= 0, "must be non-negative"

        denominator = num_items * sum_of_squared_count

        assert numerator >= 0, "numerator must be non-negative"
        assert denominator > 0, "denominator must be positive"

        jain_index = numerator/denominator

        rounded_jain = round(jain_index,15)
        assert approx(jain_index) == approx(rounded_jain)
        jain_index = rounded_jain

        assert jain_index >= 0 and jain_index <=1, "need to be non-negative and not more than 1"

        n = num_items
        jain_max = numerator / n
        jain_max /= n* (floor_km_n**2) + km_mod_n*(2*floor_km_n + 1)

        norm_jain_index = (jain_index - k/n)/(jain_max - k/n)
        rounded_jain_our = round(norm_jain_index,15)
        assert approx(norm_jain_index) == approx(rounded_jain_our)
        norm_jain_index = rounded_jain_our

        if approx(norm_jain_index) == 0:
            norm_jain_index = 0
        elif approx(norm_jain_index) == 1:
            norm_jain_index = 1
        else:
            assert norm_jain_index >= 0 and norm_jain_index <=1, "need to be non-negative and not more than 1"


        assert abs(norm_jain_index) == norm_jain_index

        return jain_index, abs(norm_jain_index) #because if it passes the assertion, it is 0, but it appears as -0.0

    def get_QF(self, item_matrix, slot, num_items, k):
        unique_count = np.unique(item_matrix).size
        qf_ori = unique_count / num_items

        numerator =  unique_count - k
        if slot >= num_items:
            denom =  num_items - k
        else:
            m = slot/k
            denom = k*(m-1)

        qf_our = numerator/denom

        return qf_ori, qf_our