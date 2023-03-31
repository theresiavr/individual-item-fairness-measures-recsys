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
            gini_ori, gini_our, gini_w_ori = self.get_gini(item_matrix_k,item_count,slot,num_items, k, km_mod_n)
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



        return gini_index, abs(gini_our), gini_w_ori

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

class MDG_OD(AbstractMetric):
    #but instead considering new items, we consider non-new but recommended items too
    metric_type = EvaluatorType.RANKING
    metric_need = ['rec.items', 'data.num_items']

    def __init__(self, config):
        super().__init__(config)
        self.topk = config['topk']
        self.t = 10

    def used_info(self, dataobject):
        """Get the matrix of recommendation items and number of items in total item set"""
        item_matrix = dataobject.get('rec.items')
        num_items = dataobject.get('data.num_items') - 1
        rec_mat = dataobject.get('rec.topk')
        pos_items = dataobject.get("data.pos_items")

        topk_idx, _ = torch.split(rec_mat, [max(self.topk), 1], dim=1)
        topk_idx = topk_idx.to(torch.bool).numpy()
        return item_matrix.numpy(), num_items, topk_idx, pos_items

    def calculate_metric(self, dataobject):
        item_matrix, num_items, topk_idx, pos_items = self.used_info(dataobject)
        metric_dict = {}

        for k in self.topk:
            mdg_ori, mdg_our, OD_E, OD_F, OD_E_our, OD_F_our = self.get_metrics(item_matrix[:, :k], num_items, topk_idx[:, :k], self.t, pos_items)

            key = '{}@{}'.format('MDG_ori', k)
            metric_dict[key] = round(mdg_ori, self.decimal_place)

            key = '{}@{}'.format('MDG_our', k)
            metric_dict[key] = round(mdg_our, self.decimal_place)

            key = '{}@{}'.format('OD-E_ori', k)
            metric_dict[key] = round(OD_E, self.decimal_place)

            key = '{}@{}'.format('OD-F_ori', k)
            metric_dict[key] = round(OD_F, self.decimal_place)

            key = '{}@{}'.format('OD-E_our', k)
            metric_dict[key] = round(OD_E_our, self.decimal_place)

            key = '{}@{}'.format('OD-F_our', k)
            metric_dict[key] = round(OD_F_our, self.decimal_place)
        return metric_dict
        

    def get_metrics(self, item_matrix, num_items, topk_idx, t, pos_items):
        def _get_mdg_i_and_merit(item):
            rank_weight = 1/np.log2(np.where(item_matrix==item)[1]+2)
            rel_indicator = topk_idx[np.where(item_matrix==item)]
            ex_i = rank_weight.sum()
            fb_i = (rank_weight * rel_indicator).sum()
            u_plus = np.count_nonzero(rel_indicator)
            mdg_i = fb_i/u_plus
            if u_plus == 0:
                mdg_i_our = 0
            else:
                mdg_i_our = mdg_i
            

            merit_our = rel_indicator.sum()
            merit_i = merit_our
           
            if merit_our == 0:
                merit_our = -1
            
            return mdg_i, merit_i, ex_i, fb_i, mdg_i_our, merit_our

        rel = pos_items
        u_i_star = Counter(np.hstack(rel.flatten()))

        list_mdg_merit_i = [_get_mdg_i_and_merit(item) for item in range(1,num_items+1)]
       
        list_mdg_i = np.asarray([x[0] for x in list_mdg_merit_i])
        list_mdg_exc_i =  np.asarray([x[4] for i,x in enumerate(list_mdg_merit_i) if u_i_star[i+1]>0])

        list_ex_i = [x[2]/x[1] for x in list_mdg_merit_i]
        list_ex_i_exc =  [x[2]/x[5] for x in list_mdg_merit_i if x[5]>0]
        list_ex_i = np.asarray(list_ex_i, dtype=float).reshape(-1,1) 
        list_ex_i_exc = np.asarray(list_ex_i_exc,dtype=float).reshape(-1,1) 

        list_fb_i = np.asarray([x[3]/x[1] for x in list_mdg_merit_i])
        list_fb_i_exc =  [x[3]/x[5] for x in list_mdg_merit_i if x[5]>0]
        list_fb_i = np.asarray(list_fb_i,dtype=float).reshape(-1,1) 
        list_fb_i_exc = np.asarray(list_fb_i_exc,dtype=float).reshape(-1,1) 

        #get relevant item matrix and unique count of relevant
        mdg_i_t = list_mdg_i[list_mdg_i <= np.percentile(list_mdg_i, t)]
        mdg_ori = mdg_i_t.mean()

        mdg_i_t_exc = list_mdg_exc_i[list_mdg_exc_i <= np.percentile(list_mdg_exc_i, t)]
        mdg_our =  mdg_i_t_exc.mean()

        #do OD
        OD_E = pdist(list_ex_i, lambda u, v: (np.abs(u-v)).sum()).mean()
        OD_F = pdist(list_fb_i, lambda u, v: (np.abs(u-v)).sum()).mean()
        OD_E_our = pdist(list_ex_i_exc, lambda u, v: (np.abs(u-v)).sum()).mean()
        OD_F_our = pdist(list_fb_i_exc, lambda u, v: (np.abs(u-v)).sum()).mean()

        return mdg_ori, mdg_our, OD_E, OD_F, OD_E_our, OD_F_our

class IAA_true(AbstractMetric): #THIS IS MEANT FOR THE ARTIFICIAL INSERTION EXPERIMENT
    metric_type = EvaluatorType.RANKING
    metric_need = ['num.items', 'data.pos_items',"rec.items"]
    smaller = True
    def __init__(self, config):
        super().__init__(config)
        self.topk = config['topk']

    def used_info(self, dataobject):
        """Get the matrix of recommendation items and number of items in total item set"""
        num_items = dataobject.get('data.num_items') - 1
        pos_items =  dataobject.get("data.pos_items")
        item_matrix = dataobject.get('rec.items')

        return num_items, pos_items, item_matrix

    def calculate_metric(self, dataobject):
        num_items, pos_items, item_matrix = self.used_info(dataobject)
        metric_dict = {}
        for k in self.topk:
            IAA_true_ori, IAA_our = self.get_IAA(num_items, pos_items, item_matrix[:, :k], k)
            key = '{}@{}'.format('IAA_true_ori', k)
            metric_dict[key] = round(IAA_true_ori, self.decimal_place)

            key = '{}@{}'.format('IAA_our', k)
            metric_dict[key] = round(IAA_our, self.decimal_place)
        return metric_dict

    def get_IAA(self, num_items, pos_items, item_matrix, k):
        #TRUE rel - ORI
        #rel part
        num_users = item_matrix.shape[0]
        all_item_ids =  np.arange(num_items) + 1 #item id start 1

        not_in_item_matrix = np.array([all_item_ids[~np.in1d(all_item_ids, curr_rec_user)] for curr_rec_user in item_matrix], dtype=int)

        full_rec_mat = np.concatenate([item_matrix, not_in_item_matrix], axis=1)
        rel = np.array([np.in1d(full_rec_mat[i], pos_items[i], assume_unique=True) for i in range(num_users)], dtype=int) 
        #does not need + 1 anymore like the actual IAA because we use the actual index.
       
        att = np.zeros_like(full_rec_mat, dtype='float64')
        att[:,:k] = (k-np.arange(1,k+1))/(k-1) #normalised
       
        IAA_u_true = np.abs(att-rel).sum(axis=1) / num_items 
        IAA_ori_true = IAA_u_true.mean()

        #TRUE rel - OUR

        num_r_u_star = rel.sum(axis=1)
        len_for_eval_our = np.where(num_r_u_star>k, num_r_u_star, k) #get max between num_r_u_star and k

        fixed_att = np.zeros_like(full_rec_mat, dtype='float64')
        fixed_att[:,:k] = (k+1-np.arange(1,k+1))/k #normalised
        
        cut_att = np.asarray([att_user[:len_for_eval_our[i]] for i, att_user in enumerate(fixed_att)], dtype=object)
        cut_rel = np.asarray([rel_user[:len_for_eval_our[i]] for i, rel_user in enumerate(rel)], dtype=object)

        try:
            IAA_u_our = np.abs(cut_att-cut_rel).sum(axis=1) / num_items 
        except:
            IAA_u_our = np.fromiter(map(sum, np.abs(cut_att-cut_rel)), float) / num_items
        IAA_our = IAA_u_our.mean()

        return IAA_ori_true, IAA_our

class IAA(AbstractMetric):
    metric_type = EvaluatorType.RANKING
    metric_need = ['data.num_items','rec.score']
    smaller = True
    def __init__(self, config):
        super().__init__(config)
        self.topk = config['topk']

    def used_info(self, dataobject):
        """Get the matrix of recommendation items and number of items in total item set"""
        num_items = dataobject.get('data.num_items') - 1
        pred_rel = dataobject.get('rec.score')
        pos_items =  dataobject.get("data.pos_items")

        try:
            removed_pred_rel = dataobject.get("rec.removed_pred_rel")
            return num_items, pred_rel[:,1:], pos_items, removed_pred_rel #pred_rel from 1 onwards to remove extra item in front
        except:
            return num_items, pred_rel[:,1:], pos_items, "" #pred_rel from 1 onwards to remove extra item in front

    def calculate_metric(self, dataobject):
        num_items, pred_rel, pos_items, removed_pred_rel = self.used_info(dataobject)
        metric_dict = {}
        for k in self.topk:

            IAA_pred_ori, IAA_true_ori = self.get_IAA(num_items, pred_rel, pos_items, k, removed_pred_rel)
            key = '{}@{}'.format('IAA_pred_ori', k)
            metric_dict[key] = round(IAA_pred_ori, self.decimal_place)

            key = '{}@{}'.format('IAA_true_ori', k)
            metric_dict[key] = round(IAA_true_ori, self.decimal_place)

        return metric_dict

    def get_IAA(self, num_items, pred_rel, pos_items, k, removed_pred_rel):
        #ORI PRED
        #relevance part
        sorted_pred_rel = pred_rel.sort(descending=True).values

        if removed_pred_rel != "":
            idx_99999 = torch.where(sorted_pred_rel==-99999)

            #replace -9999 with removed pred rel
            sorted_pred_rel[idx_99999] = removed_pred_rel.reshape_as(sorted_pred_rel[idx_99999])

        nan_val = sorted_pred_rel * (sorted_pred_rel != -np.inf).to(torch.int)

        #https://stackoverflow.com/questions/18689235/numpy-array-replace-nan-values-with-average-of-columns row-wise minimum value
        filled_pred_rel = np.where(np.isnan(nan_val), ma.array(nan_val, mask=np.isnan(nan_val)).min(axis=1)[:,np.newaxis], nan_val)

        #normalisation
        min_val = filled_pred_rel.min(axis=1)[:, np.newaxis]
        max_val = filled_pred_rel.max(axis=1)[:, np.newaxis]

        denominator = max_val-min_val
        denominator[denominator==0] = denominator[denominator==0] +1  #avoid zero division due to rel prediction as 0
        rel = (filled_pred_rel-min_val)/denominator

        #attention part
        att = np.zeros_like(pred_rel, dtype='float64')
        att[:,:k] = (k-np.arange(1,k+1))/(k-1) #normalised

        IAA_u = np.abs(att-rel).sum(axis=1) / num_items 
        IAA_ori_pred = IAA_u.mean()

        #TRUE rel - ORI
        #rel part
        full_rec_mat = pred_rel.sort(descending=True, stable=True).indices.numpy()
        rel = np.array([np.in1d(full_rec_mat[i]+1, pos_items[i], assume_unique=True) for i in range(pos_items.size)], dtype=int) 
        #rec_mat need +1 because item index in pos_items start from 1 instead of 0
        #pos_items size because it is np array of type object (different length), true IAA cannot use size but shape[0] instead
        IAA_u_true = np.abs(att-rel).sum(axis=1) / num_items 
        IAA_ori_true = IAA_u_true.mean()

        return IAA_ori_pred, IAA_ori_true


class MME_IIF_AIF(AbstractMetric):
    metric_type = EvaluatorType.RANKING
    metric_need = ['rec.items', 'data.num_items', "data.pos_items"]
    smaller = True
    def __init__(self, config):
        super().__init__(config)
        self.topk = config['topk']

    def used_info(self, dataobject):
        """Get the matrix of recommendation items and number of items in total item set"""
        
        item_matrix = dataobject.get('rec.items')
        num_items = dataobject.get('data.num_items') - 1
        pos_items = dataobject.get('data.pos_items')

        return item_matrix.numpy(), num_items, pos_items

    def calculate_metric(self, dataobject):
        item_matrix, num_items, pos_items = self.used_info(dataobject)
        metric_dict = {}
        for k in self.topk:
            MME, II_A, AI_F = self.get_metrics(item_matrix[:, :k], num_items, pos_items)

            key = '{}@{}'.format('MME_ori', k)
            metric_dict[key] = round(MME, self.decimal_place)

            key = '{}@{}'.format('II-F_ori', k)
            metric_dict[key] = round(II_A, self.decimal_place)

            key = '{}@{}'.format('AI-F_ori', k)
            metric_dict[key] = round(AI_F, self.decimal_place)

        return metric_dict

    def get_metrics(self, item_matrix, num_items, pos_items):
        rec = item_matrix
        rel = pos_items
        m = rec.shape[0]

        gamma = 0.8

        #build user-item matrix
        user_item_rel = np.zeros((m, num_items))


        rank_matrix = np.zeros_like(user_item_rel)
        for i in range(len(rec)):
            rank_matrix[i][rec[i]-1] = np.where(rec[i])[0]+1
        
        user_item_exp_inv = np.copy(rank_matrix)
        user_item_exp_rbp = np.copy(rank_matrix)

        user_item_exp_inv[user_item_exp_inv.nonzero()] = 1/user_item_exp_inv[user_item_exp_inv.nonzero()]
        user_item_exp_rbp[user_item_exp_rbp.nonzero()] = gamma**(user_item_exp_rbp[user_item_exp_rbp.nonzero()]-1)

        for i in range(len(rel)):
            user_item_rel[i][rel[i]-1] = 1

        max_envies = np.zeros(num_items)

        #Credits to https://github.com/usaito/kdd2022-fair-ranking-nsw/blob/main/src/synthetic/func.py#L64
        for i in range(num_items):
            u_d_swap = (user_item_exp_inv * user_item_rel[:, [i]*num_items]).sum(0)
            d_envies = u_d_swap - u_d_swap[i]
            max_envies[i] = d_envies.max()

        MME = max_envies.mean() / m

        #start II-F and AI-F
        r_u_star = user_item_rel.sum(1)[:,np.newaxis]

        e_ui_star = user_item_rel  * (1-np.power(gamma, r_u_star))/(1-gamma)
        r_u_star[r_u_star==0] = 1

        e_ui_star /= r_u_star

        diff = user_item_exp_rbp - e_ui_star
        II_F = np.power(diff, 2).mean()
        AI_F = np.power(diff.mean(0),2).mean()

        return MME, II_F, AI_F

class IBOIWO(AbstractMetric):
    metric_type = EvaluatorType.RANKING
    metric_need = ['rec.items', 'data.num_items', "data.pos_items"]

    def __init__(self, config):
        super().__init__(config)
        self.topk = config['topk']

    def used_info(self, dataobject):
        """Get the matrix of recommendation items and number of items in total item set"""
        
        item_matrix = dataobject.get('rec.items')
        num_items = dataobject.get('data.num_items') - 1
        pos_items = dataobject.get('data.pos_items')

        return item_matrix.numpy(), num_items, pos_items

    def calculate_metric(self, dataobject):
        item_matrix, num_items, pos_items = self.used_info(dataobject)
        metric_dict = {}

        for k in self.topk:
            
            IBO_ori, IWO_ori, IBO_our, IWO_our = self.get_IBOIWO(item_matrix[:, :k], num_items, pos_items)
            key = '{}@{}'.format('IBO_ori', k)
            metric_dict[key] = round(IBO_ori, self.decimal_place)

            key = '{}@{}'.format('IWO_ori', k)
            metric_dict[key] = round(IWO_ori, self.decimal_place)

            key = '{}@{}'.format('IBO_our', k)
            metric_dict[key] = round(IBO_our, self.decimal_place)

            key = '{}@{}'.format('IWO_our', k)
            metric_dict[key] = round(IWO_our, self.decimal_place)

        return metric_dict

    def get_IBOIWO(self, item_matrix, num_items, pos_items):
        rec = item_matrix
        rel = pos_items
        m = rec.shape[0]
        k = rec.shape[1]
        inv = 1/(np.arange(k, dtype="int")+1)


        user_item_rel = np.zeros((m, num_items))
        rank_matrix = np.zeros_like(user_item_rel)
        for i in range(len(rec)):
            rank_matrix[i][rec[i]-1] = np.where(rec[i])[0]+1
        
        user_item_exp_inv = np.copy(rank_matrix)
        user_item_exp_inv[user_item_exp_inv.nonzero()] = 1/user_item_exp_inv[user_item_exp_inv.nonzero()]

        for i in range(len(rel)):
            user_item_rel[i][rel[i]-1] = 1

        imp_i_arr = (user_item_exp_inv* user_item_rel).sum(0)/m
        imp_unif_arr = user_item_rel.sum(0) * inv.sum()/num_items/m

        ratio = imp_i_arr/imp_unif_arr
        if any(np.isnan(ratio)):
            IBO_ori = np.nan
            IWO_ori = np.nan
        else:
            better_ori = ratio >= 1.1
            worse_ori = ratio <= 0.9
            IBO_ori = better_ori.sum()/num_items
            IWO_ori = worse_ori.sum()/num_items

        mask = imp_unif_arr!=0
        filtered_imp_unif = imp_unif_arr[mask] #only items that have at least 1 relevant user

        filtered_imp_i = imp_i_arr[mask]

        better_our = filtered_imp_i >= (1.1 * filtered_imp_unif)
        worse_our = filtered_imp_i <= (0.9 * filtered_imp_unif)
        
        IBO_our = better_our.mean()
        IWO_our = worse_our.mean()

        return IBO_ori, IWO_ori, IBO_our, IWO_our 