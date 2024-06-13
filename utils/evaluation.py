import sys

def mean_average_precision(sort_data):
    count_1 = 0
    sum_precision = 0
    for index in range(len(sort_data)):
        if sort_data[index][1] == 1:
            count_1 += 1
            sum_precision += 1.0 * count_1 / (index+1)
    return sum_precision / count_1

def get_p_at_n_in_m(data, n, m, ind):
    pos_score = data[ind][0]
    curr = data[ind:ind+m]
    curr = sorted(curr, key=lambda x: x[0], reverse=True)

    if curr[n-1][0] <= pos_score:
        return 1
    return 0

def evaluate(file_path):
    data = []
    x=0
    total_num=0
    sum_m_a_p=0
    m_a_p=0
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            tokens = line.split("\t")
            
            if len(tokens) != 2:
                continue
            
            data.append((float(tokens[0]), int(tokens[1])))
            if x % 10 == 9:
                total_num += 1
                total_num += 1
                m_a_p= mean_average_precision(data)
                sum_m_a_p += m_a_p
            x += 1
                
    p_at_1_in_2 = 0.0
    p_at_1_in_10 = 0.0
    p_at_2_in_10 = 0.0
    p_at_5_in_10 = 0.0

    length = int(len(data)/10)
    
    for i in range(0, length):
        ind = i * 10
        assert data[ind][1] == 1
        
        p_at_1_in_2 += get_p_at_n_in_m(data, 1, 2, ind)
        p_at_1_in_10 += get_p_at_n_in_m(data, 1, 10, ind)
        p_at_2_in_10 += get_p_at_n_in_m(data, 2, 10, ind)
        p_at_5_in_10 += get_p_at_n_in_m(data, 5, 10, ind)

    return (p_at_1_in_2/length, p_at_1_in_10/length, p_at_2_in_10/length, p_at_5_in_10/length, 1.0*sum_m_a_p/total_num)
