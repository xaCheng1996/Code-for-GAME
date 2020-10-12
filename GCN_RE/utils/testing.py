import logging
_logger = logging.getLogger(__name__)

def get_chunks(l, n):
    return [l[i:i + n] for i in range(0, len(l), n)]


def bin_data_into_buckets(data, batch_size):
    buckets = []
    chunks = get_chunks(data, batch_size)
    for chunk in chunks:
        buckets.append(chunk)
    return buckets

def get_gcn_results(gcn_model, data, maxlength,batch_size, RE_filename, threshold):

    import GCN_RE.utils.auxiliary as aux
    '''
    words, word_vector, word_isEntity, word_index,  relation_entity, relation_vector
    '''
    batch_cnt = 0
    batches = bin_data_into_buckets(data, batch_size=batch_size)
    item_all = 0
    TP_all = 0
    FP_all = 0
    FN_all = 0
    all_batch = len(batches)
    for batch in batches:
        bucket_data = aux.get_data_from_sentences(batch, maxlength)
        # for item in batch:
        #     '''
        #    word_list, word_vector, subj_start, subj_end, obj_start,obj_end, relation_vector
        #    '''
        #     # print(len(item))
        #     words = item[0]
        #     word_embeddings = item[1]
        #     subj_start = item[2]
        #     subj_end = item[3]
        #     obj_start = item[4]
        #     obj_end = item[5]
        #     relation_vector = item[6]
        #     relation_vector = [np.array(relation_vector)]
        #     edges = item[7]
        #     A_fw, A_bw, X = \
        #         aux.create_graph_from_sentence_and_word_vectors(words, word_embeddings, subj_start, subj_end,
        #                                                         obj_start, obj_end, edges)
        #     # print(word_is_entity)
        #     # print(value_matrix_1)
        #     gcn_batch.append((A_fw, A_bw, X, relation_vector, subj_start, subj_end, obj_start, obj_end))
        #     # print(gcn_bucket)
        #     # print(len(gcn_bucket))

        if len(bucket_data) >= 1:
            TP, FN ,FP = gcn_model.predict(data=bucket_data, RE_filename = RE_filename, threshold = threshold)
            # if cnt_batch % 1 == 0:
            if TP == 0 or FN == 0 or FP ==0:
                TP += 1
                FN += 1
                FP += 1
            P = float(TP)/float(TP+FP)
            R = float(TP)/float(TP+FN)
            if P == 0 or R == 0:
               F = 0
            else: F = 2*R*P/(R+P)
            import time
            now = time.strftime("%H:%M:%S")
            if batch_cnt%100 == 0:
                print('{}, This Step is Batch {}/{}, P:{:.2f}, R:{:.2f}, F:{:.2f}'.
                      format(now, batch_cnt, all_batch, P, R, F))
            item_all += len(bucket_data)
            batch_cnt += 1
            TP_all += TP
            FN_all += FN
            FP_all += FP
    P = float(TP_all) / float(TP_all + FP_all)
    R = float(TP_all) / float(TP_all + FN_all)
    F = 2 * R * P / (R + P)
    return P,R,F
