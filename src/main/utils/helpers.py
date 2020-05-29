import pandas

def unify_registed_client_end(s):
    if s is None:
        return 0
    s = s.strip().lower()
    if s == 'ios':
        return 3
    elif s == 'android':
        return 2
    elif s == 'h5':
        return 1
    else:
        return 0


def unify_phone_brand(s):
    if s is None:
        return 'OTH'
    s = str(s).strip().upper()
    if s.startswith('AGM'):
        return 'AGM'
    elif s.startswith('AUM'):
        return 'AUM'
    elif s.startswith('A'):
        return 'A'
    elif s.startswith('BLA'):
        return 'BLA'
    elif s.startswith('BTV'):
        return 'BTV'
    elif s.startswith('BLN') or s.startswith('BND'):
        return 'OTH'
    elif s.startswith('B'):
        return 'B'
    elif s.startswith('COR'):
        return 'COR'
    elif s.startswith('C'):
        return 'C'
    elif s.startswith('HUAWEI'):
        return 'HUAWEI'
    elif s.startswith('MI'):
        return 'MI'
    elif s.startswith('SM'):
        return 'SM'
    elif s.startswith('M5'):
        return 'M5'
    elif s.startswith('OPPO') or 'OPPO' in s:
        return 'OPPO'
    elif s.startswith('I') or 'IPHONE' in s:
        return 'I'
    elif s.startswith('VIVO') or 'VIVO' in s:
        return 'VIVO'
    elif s.startswith('NX'):
        return 'NX'
    else:
        return 'OTH'


def unify_education(s):
    if s is None:
        return 4
    if '中专' in s:
        return 1
    if '初中' in s:
        return 0
    if '高职' in s:
        return 2
    if '本科' in s:
        return 3
    else:
        return 4


def calculate_profit(pp, pf, fp, ff):
    good_ppl_origin = pp + pf
    bad_ppl = fp + ff
    num_ppl_in_total = good_ppl_origin + bad_ppl
    good_ppl_origin_ratio = good_ppl_origin / num_ppl_in_total
    good_ppl_optimized_ratio = pp / (fp + pp)
    good_ppl_optimize_picking_ratio = (pp + fp) / num_ppl_in_total
    i = 0
    num_need_to_select_origin = 10000
    num_need_to_select_optimized = 10000
    # print('Number of ppl in total: ', num_ppl_in_total, ' good ppl: ', good_ppl_origin)
    good_ppl_earning = 300
    cost_per_person = 200
    bad_ppl_lost = 1200

    i = 0
    earning_origin = 0
    earning_optimized = 0
    while (i < 52):
        earning_this_round_origin = (
                                            10000 - num_need_to_select_origin) * good_ppl_earning + num_need_to_select_origin * good_ppl_origin_ratio * good_ppl_earning - cost_per_person * num_need_to_select_origin - bad_ppl_lost * num_need_to_select_origin * (
                                            1 - good_ppl_origin_ratio)
        print("earning this round origin: ", earning_this_round_origin)
        num_need_to_select_origin = num_need_to_select_origin - num_need_to_select_origin * good_ppl_origin_ratio * 0.7
        earning_this_round_optimized = (
                                               10000 - num_need_to_select_optimized) * 500 + num_need_to_select_optimized * good_ppl_optimized_ratio * good_ppl_earning - cost_per_person * num_need_to_select_optimized / good_ppl_optimize_picking_ratio - bad_ppl_lost * num_need_to_select_optimized * (
                                               1 - good_ppl_optimized_ratio)
        print("earning this round optimized: ", earning_this_round_optimized)
        num_need_to_select_optimized = num_need_to_select_optimized - num_need_to_select_optimized * good_ppl_optimized_ratio * 0.7
        i = i + 1
        earning_origin = earning_origin + earning_this_round_origin
        earning_optimized = earning_optimized + earning_this_round_optimized

    return earning_optimized - earning_origin


def new_profit_cal(pp, pf, fp, ff):
    origin_ratio = (pp + pf) / (pp + pf + fp + ff)
    new_ratio = (pp) / (pp + fp)
    total_ppl = 10000
    origin_earning = total_ppl * origin_ratio * 300 * 4 - 300 * total_ppl - 1200 * total_ppl * (1 - origin_ratio)
    new_earning = total_ppl * new_ratio * 300 * 4 - 300 * total_ppl / (
            (pp + fp) / (pp + pf + fp + ff)) - 1200 * total_ppl * (1 - new_ratio)
    return new_earning - origin_earning


def customize_acc(y_true, y_pred):
    count_p_p = 0
    count_p_f = 0
    count_f_f = 0
    count_f_p = 0
    if y_true is None or y_pred is None:
        print('null input')
    elif len(y_pred) != len(y_true):
        print('length no equal: ', len(y_true), '  ', len(y_pred))
    else:
        for i in range(0, len(y_true)):
            if y_true[i] == 0:
                if y_pred[i] == 0:
                    count_p_p = count_p_p + 1
                else:
                    count_p_f = count_p_f + 1
            else:
                if y_pred[i] == 0:
                    count_f_p = count_f_p + 1
                else:
                    count_f_f = count_f_f + 1
        if ((count_p_p + count_f_p) == 0):
            return None
        return new_profit_cal(count_p_p, count_p_f, count_f_p, count_f_f), count_p_p, count_p_f, count_f_p, count_f_f


def ensemble(results):
    ensembled_result = []
    for i in range(0, len(results[0])):
        count = 0
        for result in results:
            count = count + result[i]
        if count > len(results) / 2:
            ensembled_result.append(1)
        else:
            ensembled_result.append(0)
    return ensembled_result


def transform_request_to_df(request):
    json_list  = request.get_json()
    df = pandas.DataFrame.from_dict(json_list, orient='columns')
    return df


def generate_X_y_from_df(df):
    y = df['好/坏（1：坏）'].values
    df.drop(columns=['好/坏（1：坏）'], inplace=True)
    X = df.values.astype(int)
    return X, y


def _generate_test_json():
    df = pandas.read_csv('/var/qindom/riskcontrol/data/jan_data.csv', sep=',',encoding='utf-8')
    df = df.head(3)
    print(df.to_json(orient='records',force_ascii=False))
