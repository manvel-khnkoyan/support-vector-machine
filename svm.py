
from random import randint
import math

def liner_equation(x,y):
    return -2*x - 3*y + 8


def nonliner_equation(x,y):
    return 2*y - 3*x*x + 5*x - 3


def generate_dataset(count, type="liner"):
    data = []
    for i in range(count):
        rx = float(randint(0, 6000))
        ry = float(randint(0, 6000))
        x = 5 - rx/1000
        y = 5 - ry/1000
        if type == "liner":
            value = liner_equation(x,y)
        else:
            value = nonliner_equation(x,y)
        if value > 1:
            element = [x,y,1]
        else:
            element = [x,y,-1]
        data.append(element)
    return data


def find_support_vectors(dataset, C=0.5, slack=0.5):
    distances = {}
    for i,v1 in enumerate(dataset):
        for j,v2 in enumerate(dataset):
            if i != j and v1[2] != v2[2]:
                distance = math.sqrt((v1[0] - v2[0])**2 + (v1[1] - v2[1])**2)
                if i not in distances or distance < distances[i]:
                    distances[i] = distance

    support_vectors = []
    for index,vector in enumerate(dataset):
        distance = distances[index]
        if distance > C - slack and distance < C + slack:
            support_vectors.append(vector)
    return support_vectors


def get_weight(dataset, alfa):
    w1 = 0
    w2 = 0
    for index, (x1,x2,y) in enumerate(dataset):
        w1 += y*x1*alfa[index]
        w2 += y*x2*alfa[index]
    return [w1,w2]


def get_basis(dataset, wight):
    b1 = None
    b2 = None
    for index, (x1,x2,y) in enumerate(dataset):
        b = -x1 * wight[0] - x2 * wight[1]
        if y == 1:
            if b1 is None or b < b1:
                b1 = b
            continue
        if b2 is None or b > b2:
            b2 = b
    return (b1+b2)/2


def get_axis(x_array,wight,basis):
    y = [None]*len(x_array)
    for index, x in enumerate(x_array):
        y[index] = -(x*wight[0] + basis)/wight[1]
    return y


def init_alfa(dataset):
    alfa = [0] * len(dataset)
    y1,y2 = 0,0
    for x1,x2,y in dataset:
        if y == 1:
            y1 += 1
        else:
            y2 += 1
    for index,(x1,x2,y) in enumerate(dataset):
        if y == 1:
            alfa[index] = 1.0/y1
        else:
            alfa[index] = 1.0/y2
    return alfa


def dot_product(x1,x2):
    dotproduct = 0
    for i, j in zip(x1, x2):
        dotproduct += i * j
    return dotproduct


def rbf_kernel(x1,x2, Q=1):
    x = x1[0] - x2[0]
    y = x1[1] - x2[1]
    return math.exp(-Q*(x**2 + y**2))


def get_lagrangian(dataset, alfa, type, Q):
    a1 = 0
    a2 = 0
    for value in alfa:
        a1 += value
    for index1,(x11,x12,y1) in enumerate(dataset):
        for index2,(x21,x22,y2) in enumerate(dataset):
            if type == "liner":
                a2 += y1 * y2 * alfa[index1] * alfa[index2] * dot_product([x11,x12],[x21,x22])
            if type == "rbf":
                a2 += y1 * y2 * alfa[index1] * alfa[index2] * rbf_kernel([x11,x12],[x21,x22], Q=Q)
    return a1 - 0.5 * a2


def change_alfa(dataset, alfa, index, delta):
    portion = delta / (len(alfa) - 1)
    # check
    if alfa[index] + delta < 0:
        return
    for i,(x1,x2,y) in enumerate(dataset):
        if i != index and alfa[i] - y * dataset[index][2] * portion < 0:
            return
    # change
    for i, (x1, x2, y) in enumerate(dataset):
        if i != index:
            alfa[i] -= y * dataset[index][2] * portion
    alfa[index] += delta


def learn_alfa(dataset, precision=0.0001, delta=0.001, Q=1, type="liner" ):
    alfa = init_alfa(dataset)
    alfa = coordinate_descent(dataset,alfa, precision=precision, delta=delta, Q=Q, type=type)
    return alfa


def predict(dataset,alfa,x_test, basis, type="liner"):
    value = 0
    for index, (x1,x2,y) in enumerate(dataset):
        if type == "liner":
            value += alfa[index]*y*dot_product([x1,x2],x_test)
        if type == "rbf":
            value += alfa[index]*y*rbf_kernel([x1,x2], x_test)
    if type == "liner":
        value += basis
    if value > 0:
        return 'o'
    return '+'


def detect_maximum_direction(dataset, alfa, index, delta, type, Q):
    direction = 1
    prev = get_lagrangian(dataset, alfa, type, Q)
    alfa[index] += delta
    next = get_lagrangian(dataset, alfa, type, Q)
    if prev > next:
        direction = -1
    alfa[index] -= delta
    return direction


def coordinate_descent(dataset, alfa, precision, delta, Q, type ):
    go = True
    while go:
        go = False
        for index, value in enumerate(dataset):
            # detecting direction dirst / up or down
            direction = detect_maximum_direction(dataset, alfa, index, delta, type, Q)
            next = get_lagrangian(dataset, alfa, type, Q)
            # changing alfa[index] till lagrangian will be maximum
            while True:
                prev = next
                change_alfa(dataset, alfa, index, direction * delta)
                next = get_lagrangian(dataset, alfa, type, Q)
                if (next - prev) < precision:
                    break
                go = True
    return alfa

