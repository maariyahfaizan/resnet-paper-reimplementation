from models.resnet56 import model
from utils.dataset import create_datasets
import tensorflow as tf

train_dataset, validation_dataset, test_dataset = create_datasets()

optimizer = tf.keras.optimizers.SGD(
    learning_rate=0.1,
    momentum=0.9
)

model.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.load_weights("models/resnet56.keras")

test_loss, test_accuracy = model.evaluate(test_dataset)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Error: {(1 - test_accuracy) * 100:.2f}%")

with open("results/resnet56/test_results.txt", "w") as f:
    f.write(f"Test Loss: {test_loss:.4f}\n")
    f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
    f.write(f"Test Error: {(1 - test_accuracy) * 100:.2f}%\n")