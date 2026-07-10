from tensorflow.keras.models import load_model
from utils.dataset import create_datasets

# Load the trained model
model = load_model(
    "models/plain56.keras",
    safe_mode=False
)

# Get the datasets
_, _, test_dataset = create_datasets()

# Evaluate
test_loss, test_accuracy = model.evaluate(test_dataset)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Error: {(1 - test_accuracy) * 100:.2f}%")


with open("results/plain56/test_results.txt", "w") as f:
    f.write(f"Test Loss: {test_loss:.4f}\n")
    f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
    f.write(f"Test Error: {(1 - test_accuracy) * 100:.2f}%\n")