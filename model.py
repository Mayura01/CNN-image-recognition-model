import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class ConvLayer:
    def __init__(self, num_filters, filter_size, num_channels=3, l2_lambda=0.1):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.num_channels = num_channels
        self.filters = np.random.randn(num_filters, filter_size, filter_size, num_channels) * 0.1
        self.l2_lambda = l2_lambda

    def iterate_regions(self, image):
        h, w, _ = image.shape
        for i in range(h - self.filter_size + 1):
            for j in range(w - self.filter_size + 1):
                im_region = image[i:(i + self.filter_size), j:(j + self.filter_size)]
                yield im_region, i, j

    def forward(self, input):
        self.last_input = input
        h, w, _ = input.shape
        output = np.zeros((h - self.filter_size + 1, w - self.filter_size + 1, self.num_filters))

        for im_region, i, j in self.iterate_regions(input):
            for f in range(self.num_filters):
                output[i, j, f] = np.sum(im_region * self.filters[f])

        return output

    def backward(self, d_L_d_out, learn_rate):
        d_L_d_filters = np.zeros(self.filters.shape)

        for im_region, i, j in self.iterate_regions(self.last_input):
            for f in range(self.num_filters):
                d_L_d_filters[f] += d_L_d_out[i, j, f] * im_region

        self.filters = (1 - learn_rate * self.l2_lambda) * self.filters - learn_rate * d_L_d_filters
        return d_L_d_out

class MaxPool2:
    def iterate_regions(self, image):
        h, w, _ = image.shape
        new_h = h // 2
        new_w = w // 2

        for i in range(new_h):
            for j in range(new_w):
                im_region = image[(i * 2):(i * 2 + 2), (j * 2):(j * 2 + 2)]
                yield im_region, i, j

    def forward(self, input):
        self.last_input = input
        h, w, num_filters = input.shape
        output = np.zeros((h // 2, w // 2, num_filters))

        for im_region, i, j in self.iterate_regions(input):
            output[i, j] = np.amax(im_region, axis=(0, 1))

        return output

    def backward(self, d_L_d_out):
        d_L_d_input = np.zeros(self.last_input.shape)

        for im_region, i, j in self.iterate_regions(self.last_input):
            h, w, f = im_region.shape
            for i2 in range(h):
                for j2 in range(w):
                    for f2 in range(f):
                        if im_region[i2, j2, f2] == np.amax(im_region):
                            d_L_d_input[i * 2 + i2, j * 2 + j2, f2] = d_L_d_out[i, j, f2]

        return d_L_d_input

class FCLayer:
    def __init__(self, input_len, output_len, l2_lambda=0.1):
        self.weights = np.random.randn(input_len, output_len) * 0.1
        self.biases = np.zeros(output_len)
        self.l2_lambda = l2_lambda

    def forward(self, input):
        self.last_input_shape = input.shape
        input = input.flatten()
        self.last_input = input
        input = input.reshape((1, -1))
        return np.dot(input, self.weights) + self.biases

    def backward(self, d_L_d_out, learn_rate):
        d_L_d_weights = np.dot(self.last_input.reshape(-1, 1), d_L_d_out)
        d_L_d_biases = d_L_d_out.mean(axis=0)
        d_L_d_input = np.dot(d_L_d_out, self.weights.T)

        self.weights = (1 - learn_rate * self.l2_lambda) * self.weights - learn_rate * d_L_d_weights
        self.biases -= learn_rate * d_L_d_biases

        return d_L_d_input.reshape(self.last_input_shape)

class CNN:
    def __init__(self, num_classes=2, l2_lambda=0.1):
        self.conv = ConvLayer(8, 3, 3, l2_lambda)
        self.pool = MaxPool2()
        self.fc = FCLayer(13 * 13 * 8, num_classes, l2_lambda)

    def forward(self, image):
        output = self.conv.forward((image / 255.0) - 0.5)
        output = self.pool.forward(output)
        output = self.fc.forward(output)
        output = sigmoid(output)
        return output

    def train(self, X, Y, epochs=1, learn_rate=0.01, validation_data=None):
        train_losses = []
        val_losses = []
        val_accuracies = []

        for epoch in range(epochs):
            total_loss = 0
            for i in range(len(X)):
                out = self.forward(X[i])
                loss = np.mean(np.square(out - Y[i]))
                total_loss += loss

                # Backpropagation
                d_loss = 2 * (out - Y[i])
                d_loss = self.fc.backward(d_loss, learn_rate)
                d_loss = self.pool.backward(d_loss)
                d_loss = self.conv.backward(d_loss, learn_rate)

            train_loss = total_loss / len(X)
            train_losses.append(train_loss)

            if validation_data:
                val_images, val_labels = validation_data
                val_loss, val_acc = self.evaluate(val_images, val_labels)
                val_losses.append(val_loss)
                val_accuracies.append(val_acc)

                print(f"Epoch {epoch + 1}/{epochs}, Loss: {round(train_loss, 2)}, Val Loss: {round(val_loss, 2)}, Val Acc: {round(val_acc, 2)}")
            else:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {round(train_loss, 2)}")

        plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
        if validation_data:
            plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
            plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.savefig('training_plot.png')

    def evaluate(self, X, Y):
        total_loss = 0
        correct = 0
        for i in range(len(X)):
            out = self.forward(X[i])
            total_loss += np.mean(np.square(Y[i] - out))
            if np.argmax(out) == Y[i]:
                correct += 1
        return total_loss / len(X), correct / len(X)

    def predict(self, X):
        predictions = []
        for i in range(len(X)):
            out = self.forward(X[i])
            predictions.append(np.argmax(out))
        return predictions

    def generate(self, X):
        out = self.forward(X)
        return np.argmax(out)

# Load dataset
dataset_dir = 'dataset/train'
dataset_test_dir = 'dataset/test'
classes = ['cats', 'dogs']
train_images = []
train_labels = []
test_images = []
test_labels = []

# Load training images
for class_idx, class_name in enumerate(classes):
    class_dir = os.path.join(dataset_dir, class_name)
    for filename in os.listdir(class_dir):
        image_path = os.path.join(class_dir, filename)
        image = cv2.imread(image_path)
        resized_image = cv2.resize(image, (28, 28))
        train_images.append(resized_image)
        train_labels.append(class_idx)

train_images = np.array(train_images)
train_labels = np.array(train_labels)

# Load test images
for class_idx, class_name in enumerate(classes):
    class_dir = os.path.join(dataset_test_dir, class_name)
    for filename in os.listdir(class_dir):
        image_path = os.path.join(class_dir, filename)
        image = cv2.imread(image_path)
        resized_image = cv2.resize(image, (28, 28))
        test_images.append(resized_image)
        test_labels.append(class_idx)

test_images = np.array(test_images)
test_labels = np.array(test_labels)

# Create and train CNN
cnn = CNN()

cnn.train(train_images, train_labels, epochs=5, learn_rate=0.01, validation_data=(test_images, test_labels))
test_loss, test_acc = cnn.evaluate(test_images, test_labels)
print('Test accuracy:', round(test_acc, 2))

# Example: Predicting a single image
input_image = cv2.imread('dog.jpg')
X = cv2.resize(input_image, (28, 28))
prediction = cnn.generate(X)
predicted_animal = 'dog' if prediction == 1 else 'cat'
print("Predicted animal:", predicted_animal)
