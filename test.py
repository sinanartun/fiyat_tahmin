import joblib
import numpy as np
model = joblib.load("/Users/synan/Documents/PyCharmProjects/fiyat_tahmin/model-1.pkl")


def input_to_np_array(value1, value2, value3, value4, value5, value6, value7, value8, value9, value10, value11):

    input_array = np.array([[value1, value2, value3, value4, value5, value6, value7, value8, value9, value10, value11]])
    return input_array


input_values = input_to_np_array(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
fiyat = model.predict(input_values)
print(fiyat)