


from svm import *
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.axhline(y=0, color='silver', linestyle='--')
ax.axvline(x=0, color='silver', linestyle='--')
ax.set_title('SVM', fontdict={'fontsize': 8, 'fontweight': 'medium'})

def main():
    dataset = generate_dataset(80,type="nonliner")
    support_vectors = find_support_vectors(dataset,  C=0.5, slack=0.5)
    alfa = learn_alfa(support_vectors,type="rbf",precision=0.0001, delta=0.001)

    for index, (x1,x2,y) in enumerate(dataset):
        if y == 1:
            plt.plot(x1, x2, 'bo', markersize=4)
        else:
            plt.plot(x1, x2, 'r+', markersize=4)
    for index, (x1,x2,y) in enumerate(support_vectors):
        if y == 1:
            plt.plot(x1, x2, 'bo', markersize=10)
        else:
            plt.plot(x1, x2, 'r+', markersize=10)

    # Prediction
    for i in range(20):
        rx = float(randint(0, 6000))
        ry = float(randint(0, 6000))
        x = [5 - rx / 1000, 5 - ry / 1000]
        r = predict(support_vectors, alfa, x, 0, type="rbf")
        plt.plot(x[0], x[1], 'g' + r, markersize=8)

    plt.show()


if __name__ == "__main__":
    main()
