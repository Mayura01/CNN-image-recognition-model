import os
import cv2
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class ConvLayer:
    def __init__(self, num_filters, filter_size, num_channels=1):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.num_channels = num_channels
        self.filters = np.random.randn(num_filters, filter_size, filter_size, num_channels) / (filter_size * filter_size)

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
        h_out, w_out, _ = d_L_d_out.shape

        for im_region, i, j in self.iterate_regions(self.last_input):
            for f in range(self.num_filters):
                d_L_d_filters[f] += np.sum(d_L_d_out[i, j, f] * im_region)

        self.filters -= learn_rate * d_L_d_filters

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
        h_out, w_out, num_filters = d_L_d_out.shape

        for im_region, i, j in self.iterate_regions(self.last_input):
            h, w, f = im_region.shape
            for i2 in range(h):
                for j2 in range(w):
                    for f2 in range(f):
                        if im_region[i2, j2, f2] == np.amax(im_region):
                            d_L_d_input[i * 2 + i2, j * 2 + j2, f2] = d_L_d_out[i, j, f2]

        return d_L_d_input

class FCLayer:
    def __init__(self, input_len, output_len):
        self.weights = np.random.randn(input_len, output_len) / input_len
        self.biases = np.zeros(output_len)

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

        self.weights -= learn_rate * d_L_d_weights
        self.biases -= learn_rate * d_L_d_biases

        return d_L_d_input.reshape(self.last_input_shape)


class CNN:
    def __init__(self, num_classes=2):
        self.conv = ConvLayer(8, 3, 1)
        self.pool = MaxPool2()
        self.fc = FCLayer(13 * 13 * 8, num_classes)

    def forward(self, image):
        output = self.conv.forward((image / 255.0) - 0.5)
        output = self.pool.forward(output)
        output = self.fc.forward(output)
        return output

    def train(self, X, Y, epochs=1, learn_rate=0.01, validation_data=None):
        for epoch in range(epochs):
            total_loss = 0
            for i in range(len(X)):
                out = self.forward(X[i])
                loss = -2 * (Y[i] - out)
                loss *= sigmoid_derivative(out)

                loss = self.fc.backward(loss, learn_rate)
                loss = self.pool.backward(loss)
                loss = self.conv.backward(loss, learn_rate)

                total_loss += np.abs(loss).sum()

            if validation_data:
                val_images, val_labels = validation_data
                val_loss = self.evaluate(val_images, val_labels)
                print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(X)}, Val Loss: {val_loss}")

            else:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(X)}")

    def evaluate(self, X, Y):
        total_loss = 0
        correct = 0
        for i in range(len(X)):
            out = self.forward(X[i])
            total_loss += np.abs(-2 * (Y[i] - out)).sum()
            if np.argmax(out) == Y[i]:
                correct += 1
        return total_loss / len(X), correct / len(X)

    # def predict(self, X):
    #     predictions = []
    #     for i in range(len(X)):
    #         out = self.forward(X[i])
    #         predictions.append(np.argmax(out))
    #     return predictions
    
    def predict(self, X):
        predictions = []
        out = self.forward(X)
        predictions.append(np.argmax(out))
        return predictions
	

# Load dataset
current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(current_dir, 'dataset/train')
dataset_test_dir = os.path.join(current_dir, 'dataset/test')

classes = ['dogs', 'cats']
train_images = []
train_labels = []
test_images = []
test_labels = []

# Load training images
for class_idx, class_name in enumerate(classes):
    class_dir = os.path.join(dataset_dir, class_name)
    for filename in os.listdir(class_dir):
        image_path = os.path.join(class_dir, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        resized_image = cv2.resize(image, (28, 28))
        train_images.append(resized_image.reshape((*resized_image.shape, 1)))
        train_labels.append(class_idx)

train_images = np.array(train_images)
train_labels = np.array(train_labels)

# Load test images
for class_idx, class_name in enumerate(classes):
    class_dir = os.path.join(dataset_test_dir, class_name)
    for filename in os.listdir(class_dir):
        image_path = os.path.join(class_dir, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        resized_image = cv2.resize(image, (28, 28))
        test_images.append(resized_image.reshape((*resized_image.shape, 1)))
        test_labels.append(class_idx)

test_images = np.array(test_images)
test_labels = np.array(test_labels)

# Create and train CNN
cnn = CNN()
# cnn.train(train_images, train_labels, epochs=1, validation_data=(test_images, test_labels))
# test_loss, test_acc = cnn.evaluate(test_images, test_labels)
# print('Test accuracy:', test_acc)

# Example: Predicting a single image
input_image = cv2.imread('dog.jpg')
X = cv2.resize(input_image, (28, 28))
predictions = cnn.predict(X)
print(predictions)
predicted_animal = 'dog' if predictions[0] == 1 else 'cat'
print("Predicted animal:", predicted_animal)