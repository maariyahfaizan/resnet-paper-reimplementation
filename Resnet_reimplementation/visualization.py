import matplotlib.pyplot as plt

from utils.dataset import load_dataset

x_train, y_train, x_test, y_test, mean_image = load_dataset()

print("Training set X shape:", x_train.shape)
print("Training set Y shape:", y_train.shape)
print("Test set X shape:", x_test.shape)
print("Test set Y shape:", y_test.shape)

for i in range(2):
    plt.imshow(x_train[i].astype("uint8"))
    plt.show()

plt.imshow(mean_image / 255.0)
plt.title("Mean Image")
plt.show()