import numpy as np
from tqdm import tqdm


def display_diff(s, new_s, mask_pos):
    for i in range(len(s)):
        if i in mask_pos:
            print('[' + s[i][0] + ']', end=' ')
        else:
            print(s[i][0], end=' ')
    print('\n')


def data_mask(s, mask_num=0, mask_rate=0):
    mask_pos_list = []
    length = len(s)
    rand_list = np.random.permutation(length)
    if mask_num:
        rand_list = rand_list[0:min(mask_num, len(rand_list) - 1)]
        # print(rand_list)
    else:
        rand_list = rand_list[0:int(mask_rate * length)]
    new_s = []
    for i in range(length):
        if i in rand_list:
            continue
        else:
            new_s.append(s[i])
    return new_s, rand_list


def item_detail(pred, masked_pred, mask_pos, print_detail=False):
    i_masked = 0
    diff_word_num = 0
    if print_detail:
        for i in range(len(pred)):
            if i in mask_pos:
                print(pred[i], [])
            else:
                if pred[i] != masked_pred[i_masked]:
                    print("####", pred[i], masked_pred[i_masked])
                    diff_word_num += 1
                else:
                    print(pred[i], masked_pred[i_masked])
                i_masked += 1
        print("diff word num: ", diff_word_num)
        print('\n')
    else:
        for i in range(len(pred)):
            if i not in mask_pos:
                if pred[i] != masked_pred[i_masked]:
                    diff_word_num += 1
                i_masked += 1

    return diff_word_num


def result_diff(prediction, masked_prediction, masked_poses, sen_diff, all_pos_signi, print_detail=False):
    if (len(prediction) != len(masked_prediction)):
        print(len(prediction), len(masked_prediction))
    assert len(prediction) == len(masked_prediction)
    diff_sen_num = 0
    ave_diff_word_num = 0
    for i in range(len(prediction)):
        j_masked = 0
        for j in range(len(prediction[i])):
            if j not in masked_poses[i]:
                assert prediction[i][j][0] == masked_prediction[i][j_masked][0]
                if prediction[i][j] != masked_prediction[i][j_masked]:
                    diff_sen_num += 1
                    sen_diff[i] = True
                    diff_word_num = item_detail(
                        prediction[i], masked_prediction[i], masked_poses[i], print_detail)
                    ave_diff_word_num += diff_word_num
                    for pos in masked_poses[i]:
                        all_pos_signi[i][pos] += diff_word_num
                    break
                j_masked += 1

    return diff_sen_num, (ave_diff_word_num / diff_sen_num) if diff_sen_num != 0 else 0


def mask_result(mask_samp, mask_num, mask_rate, display_detail, eval_func, model, test_data, all_masked_test_data, param):
    prediction = eval_func(model, test_data, **param["origin_param"])
    all_pos_signi = [[0 for w in sen['words']] for sen in test_data]
    sen_diff = [False for sen in test_data]

    all_diff_sen_num = 0
    all_diff_word_num = 0
    for i in tqdm(range(mask_samp)):
        masked_prediction = eval_func(
            model, all_masked_test_data[i]['data'], **param["mask_param"])
        diff_sen_num, ave_diff_word_num = result_diff(
            prediction, masked_prediction, all_masked_test_data[i]['pos'], sen_diff, all_pos_signi, display_detail)
        # all_diff_sen_num += diff_sen_num
        all_diff_word_num += ave_diff_word_num

    for i in range(len(sen_diff)):
        if (sen_diff[i]):
            all_diff_sen_num += 1
            print("No.{}: ".format(i), all_pos_signi[i])

    print("Diff sentence num: {}/{}. Average diff word num: {}".format(
        all_diff_sen_num, len(test_data), all_diff_word_num / mask_samp))
