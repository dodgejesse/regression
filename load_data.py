import numpy as np




def get_example_data():
    #return np.asarray([[7,0.361],[8,0.299],[9,0.247],[10,0.185],[11,0.169],[12,0.139],[13,0.127],[14,0.105],[15,0.105],[16,0.095],[17,0.095],[18,0.079],[19,0.079],[20,0.079],[23,0.059],[26,0.054],[29,0.04],[38,0.037],[54,0.023]])
    #return np.asarray([[7,0.36127],[8,0.29857],[9,0.24675],[10,0.18539],[11,0.16853],[12,0.13929],[13,0.12662],[14,0.10465],[15,0.10465],[16,0.09513],[17,0.09513],[18,0.07862],[19,0.07862],[20,0.07862],[23,0.059],[29,0.04]])

    # when len(ns) == 20
    #test_nums = np.asarray([3, 4, 5, 6, 7, 9, 10, 13, 15, 18, 22, 27, 33, 39, 47, 57, 69, 83, 100])
    # when len(ns) == 30
    test_nums = np.asarray([3, 4, 5, 6, 7, 8, 10, 11, 12, 14, 16, 18, 20, 23, 26, 29, 33, 38, 42, 48, 54, 61, 69, 78, 88, 100])
    d = 5
    if d == 1:
        data = np.asarray([[3,46.65074],[4,2.94083],[5,0.93704],[6,0.52894],[7,0.36127],[8,0.29857],[9,0.24675],[10,0.18539],[11,0.16853],[12,0.13929],[13,0.12662],[14,0.10465],[15,0.10465],[16,0.09513],[17,0.09513],[18,0.07862],[19,0.07862],[20,0.07862],[22,0.06498],[23,0.05907],[26,0.0537],[29,0.04035],[38,0.03668],[54,0.02277]])


    elif d == 2:
        data = np.asarray([[3,90.90909],[4,35.04939],[5,4.30568],[6,2.94083],[7,1.66002],[8,1.2472],[9,1.13382],[10,0.93704],[11,0.93704],[12,0.64001],[13,0.70401],[14,0.64001],[15,0.58183],[16,0.52894],[17,0.52894],[18,0.43714],[19,0.48085],[22,0.32843],[26,0.32843]])
    elif d == 3:
        data = np.asarray([[3,100.0],[4,56.44739],[5,23.9392],[6,5.20987],[7,4.30568],[8,3.23492],[9,2.43044],[11,2.00863],[12,1.50911],[14,1.2472],[17,0.85186],[19,0.93704],[22,0.85186],[26,0.77441]])
    elif d == 5:
        data = np.asarray([[3,100.0],[4,100.0],[5,68.30135],[6,35.04939],[7,14.86436],[8,6.30394],[9,5.73086],[11,4.30568],[12,4.30568],[14,3.23492],[17,2.67349],[19,2.67349],[22,2.00863],[26,1.82603]])


    #test = data[21:]
    #test_nums = test_nums[test_nums > max(train[:,0])]
    test_nums = [test_nums[i] for i in range(len(test_nums)) if test_nums[i] not in data[:,0]]
    test = np.asarray([[num,0] for num in test_nums])

    train = data[4:]
    #train = data
        


    return train, test
