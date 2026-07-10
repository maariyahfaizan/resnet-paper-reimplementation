from utils.dataset import create_datasets
from models.resnet56 import model
import tensorflow as tf
import os
import json
import matplotlib.pyplot as plt

model_name = "resnet56"

optimizer = tf.keras.optimizers.SGD(
    learning_rate=0.1,
    momentum=0.9
)

train_dataset, validation_dataset, test_dataset = create_datasets()

model.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

def lr_schedule(epoch):

    if epoch < 91:
        return 0.1
    elif epoch < 136:
        return 0.01
    else:
        return 0.001


lr_callback = tf.keras.callbacks.LearningRateScheduler(
    lr_schedule
)

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=182,
    callbacks=[lr_callback]
)

model.save(f"models/{model_name}.keras")

# Create results directory
results_dir = f"results/{model_name}"
os.makedirs(results_dir, exist_ok=True)

# Save training history
with open(f"{results_dir}/history.json", "w") as f:
    json.dump(history.history, f)

# Accuracy plot
plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Train")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.title(f"{model_name} Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.savefig(f"{results_dir}/accuracy.png", dpi=300)
plt.close()

# Loss plot
plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Train")
plt.plot(history.history["val_loss"], label="Validation")
plt.title(f"{model_name} Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig(f"{results_dir}/loss.png", dpi=300)
plt.close()
