import numpy as np
import math
import re


numbers = list('0123456789')
operators = list('+-')
alphabets = [*numbers, *operators, ' ']

numbers = 3  # several number in each clause
maximum = 99  # maximum number exists at all


def random_join(array, fill_chars):
    fills = np.random.choice(fill_chars, size=len(array)-1)
    new_arr = []
    
    for value, fill in zip(array, fills):
        new_arr.append(value)
        new_arr.append(fill)
    new_arr.append(array[-1])
    
    return ''.join(map(str, new_arr))


_encoder_dict = {c: i for i, c in enumerate(alphabets)}
def encoder(array):
    return [[_encoder_dict[c] for c in pack] for pack in array]


_decoder_dict = {i: c for i, c in enumerate(alphabets)}
def decoder(array):
    return [[_decoder_dict[c] for c in pack] for pack in array]


def one_hot_encode(array, length=len(alphabets)):  # == tensorflow.keras.utils.to_categorical
    new_arr = []

    for pack in array:
        new_pack = []

        for item in pack:
            vector = [0 for _ in range(length)]
            vector[item] = 1
            new_pack.append(vector)

        new_arr.append(new_pack)
 
    return new_arr


def tensor2text(tensor):  # ~ One Hot Encoding
    texts = []

    for x in tensor:
        chars = list(map(np.argmax, x))
        result = ''.join(decoder([chars])[0])
        texts.append(result)

    return texts


def get_numbers_from_string(string, delimiters=operators + [' ']):
    pattern = '|'.join(map(lambda operator: f"\\{operator}", operators))
    numbers = re.split(pattern, string)
    numbers = list(map(lambda x: x.strip(), numbers))
    numbers = list(filter(bool, numbers))
    numbers = list(map(int, numbers))
    return numbers


def reformat_strings(array, randomize=True):
    maximum = np.max(list(map(get_numbers_from_string, array)))
    length = (len(str(maximum)) * numbers) + (numbers * 2)  # length of numbers + length of operators + blank spaces

    new_array = []
    modes = ['<', '^', '^', '>']
    
    for x in array:
        if randomize:
            mode = np.random.choice(modes)
        else:
            mode = '>'

        new_array.append(format(x, f"{mode}{length}"))

    return new_array


def string_from_numbers(X, randomize=True):
    X_str = []
    y_str = []

    fillers = [' {}', '{}', ' {} ', '{}', '{} ']
    constant_char_fills = [' {}'.format(sign) for sign in operators]
    random_char_fills = [filler.format(operator) for filler in fillers for operator in operators]
    char_fills = random_char_fills if randomize else constant_char_fills

    for pack in X:
        x = random_join(pack, char_fills)

        X_str.append(x)
        y_str.append(str(math.ceil(abs(eval(x)))))


    max_len = max(map(len, y_str))
    y_str = list(map(lambda i: format(i, f'>{max_len}'), y_str))
    X_str = reformat_strings(X_str)

    return X_str, y_str


def generate_data(clauses, maximum=maximum, numbers=numbers, randomize=True):
    X = np.random.randint(low=maximum, size=(clauses, numbers))

    X_str, y_str = string_from_numbers(X, randomize=randomize)

    X_enc = encoder(X_str)
    y_enc = encoder(y_str)

    X_ohe = one_hot_encode(X_enc)
    y_ohe = one_hot_encode(y_enc)
    
    return X_ohe, y_ohe


def predict(X: list, model):
    X = reformat_strings(X)
    X_enc = encoder(X)
    X_ohe = one_hot_encode(X_enc)

    y_ohe = model.predict(X_ohe)
    y = tensor2text(y_ohe)

    return y
