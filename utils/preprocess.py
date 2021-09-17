from skimage.transform import resize
from sklearn.model_selection import train_test_split

def preprocess(data, res):
    n_samples = len(data)
    data = resize(data, (n_samples, res, res))
    data = data.reshape((n_samples, -1)) 
    return data

def split_data(data, target, split):
    v_split = split[0]
    t_split = split[1]
    x_train, x_test, y_train, y_test = train_test_split(
        data, target, test_size=t_split+v_split, shuffle=False)

    x_val, x_test, y_val, y_test = train_test_split(
        x_test,y_test, test_size=v_split/(t_split+v_split), shuffle=False)
    print("\nNumber of samples in train:val:test = {}:{}:{}".format(len(x_train), len(x_val), len(x_test)))

    return x_train, y_train, x_test, y_test, x_val, y_val