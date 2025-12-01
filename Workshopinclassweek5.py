import tensorflow as tf
import matplotlib.pyplot as plt

# === Part 1: Load Data ===
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Printing the shapes
print("train_images shape: ", train_images.shape)
print("train_labels shape: ", train_labels.shape)
print("test_images shape: ", test_images.shape)
print("test_labels shape: ", test_labels.shape)

# Displaying first 9 images of dataset
fig = plt.figure(figsize=(10, 10))
nrows = 3
ncols = 3
for i in range(9):
    fig.add_subplot(nrows, ncols, i+1)
    plt.imshow(train_images[i])
    plt.title("Digit: {}".format(train_labels[i]))
    plt.axis(False)
plt.show()

# 2) Preprocessing the Data
# Converting image pixel values to 0 - 1
train_images = train_images / 255.0
test_images = test_images / 255.0

print("First Label before conversion:")
print(train_labels[0])

# Converting labels to one-hot encoded vectors
train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)

print("First Label after conversion:")
print(train_labels[0])

# 3) Build Neural Network Model
model = tf.keras.models.Sequential([
    # Flatten Layer that converts 2D 28x28 images to 1D array
    tf.keras.layers.Flatten(),
    
    # Hidden Layer with 512 units and ReLU activation
    tf.keras.layers.Dense(units=2, activation='relu'),
    
    # Output Layer with 10 units for 10 classes and Softmax activation
    tf.keras.layers.Dense(units=10, activation='softmax')
])

# 4) Compiling the Model
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# 5) Training a Neural Network
# Part 5: Training the model and plotting history
history = model.fit(
    x=train_images,
    y=train_labels,
    epochs=10
)

# Showing plot for loss
plt.figure()
plt.plot(history.history['loss'], label='loss')
plt.xlabel('epochs')
plt.legend(['loss'])
plt.show()

# Showing plot for accuracy
plt.figure()
plt.plot(history.history['accuracy'], color='orange')
plt.xlabel('epochs')
plt.legend(['accuracy'])
plt.show()

# 6) Evaluating a neural network
# Part 6: Evaluate on test data
test_loss, test_accuracy = model.evaluate(
    x=test_images,
    y=test_labels
)

print("Test Loss: %.4f" % test_loss)
print("Test Accuracy: %.4f" % test_accuracy)

# 7) Inference and Prediction
# Predict probabilities for test images
predicted_probabilities = model.predict(test_images)

# Predicted classes (argmax over probabilities)
predicted_classes = tf.argmax(predicted_probabilities, axis=-1).numpy()

index = 11  # example index

# Show image at index
import matplotlib.pyplot as plt
plt.imshow(test_images[index], cmap='gray')
plt.title("Index: {}".format(index))
plt.axis('off')
plt.show()

# Print probabilities for the index
print("Probabilities predicted for image at index", index)
print(predicted_probabilities[index])

# Print predicted class for the index
print("Predicted class for image at index", index, ":", predicted_classes[index])