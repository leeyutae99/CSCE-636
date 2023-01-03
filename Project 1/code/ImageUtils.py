import numpy as np
from matplotlib import pyplot as plt


"""This script implements the functions for data augmentation
and preprocessing.
"""

def parse_record(record, training):
    """ Parse a record to an image and perform data preprocessing.

    Args:
        record: An array of shape [3072,]. One row of the x_* matrix.
        training: A boolean. Determine whether it is in training mode.

    Returns:
        image: An array of shape [3, 32, 32].
    """
    # Reshape from [depth * height * width] to [depth, height, width].
    depth_major = record.reshape((3, 32, 32))

    # Convert from [depth, height, width] to [height, width, depth]
    image = np.transpose(depth_major, [1, 2, 0])

    image = preprocess_image(image, training)

    # Convert from [height, width, depth] to [depth, height, width]
    image = np.transpose(image, [2, 0, 1])

    return image


def preprocess_image(image, training):
    """ Preprocess a single image of shape [height, width, depth].

    Args:
        image: An array of shape [32, 32, 3].
        training: A boolean. Determine whether it is in training mode.
    
    Returns:
        image: An array of shape [32, 32, 3].
    """
    if training:
        ### YOUR CODE HERE
        # Resize the image to add four extra pixels on each side.
        h, w, d = image.shape
        image_padded = np.zeros((h+8,w+8,d))
        # add the original image into the padded image of pixel 0
        image_padded[4:4 + h, 4:4 + w] = image
        ### YOUR CODE HERE
        
        ### YOUR CODE HERE
        # Randomly crop a [32, 32] section of the image.
        # HINT: randomly generate the upper left point of the image
        rand_x = np.random.randint(9)
        rand_y = np.random.randint(9)
        image_cropped = image_padded[rand_y:rand_y+32, rand_x:rand_x+32]
        ### YOUR CODE HERE

        ### YOUR CODE HERE
        # Randomly flip the image horizontally.
        random_flip = np.random.randint(2)
        if random_flip == 1:
            image_cropped = np.fliplr(image_cropped)
        image = image_cropped
        ### YOUR CODE HERE

    ### YOUR CODE HERE
    np.seterr(all='ignore')
    # Subtract off the mean and divide by the standard deviation of the pixels.
    image = (image - np.mean(image, axis=(0, 1), keepdims=True)) / (np.std(image, axis=(0, 1), keepdims=True))
    image = np.nan_to_num(image, nan=0.0)
    ### YOUR CODE HERE

    return image


def visualize(image, save_name='test.png'):
    """Visualize a single test image.
    
    Args:
        image: An array of shape [3072]
        save_name: An file name to save your visualization.
    
    Returns:
        image: An array of shape [32, 32, 3].
    """
    ### YOUR CODE HERE
    img = image.reshape((3,32,32))
    img = np.transpose(img,[1,2,0])
    plt.imshow(img)
    ### YOUR CODE HERE
    plt.savefig(save_name)
    return image

# Other functions
### YOUR CODE HERE

### END CODE HERE