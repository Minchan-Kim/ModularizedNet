import tensorflow as tf


def ModularizedNet(timesteps = 5, num_joint = 7, num_data_type = 7, dropout_rate = 0.2, regularization_factor = 0.000004, learning_rate = 0.001):
    num_joint_data = (timesteps * (num_data_type - 1))
    num_input = num_joint * num_joint_data + timesteps
    inputs = tf.keras.layers.Input(shape = (num_input,))
    kernel_regularizer = tf.keras.regularizers.l2(regularization_factor)
    bias_initializer = tf.keras.initializers.RandomNormal()
    module_outputs = []
    for i in range(num_joint):
        x = tf.keras.layers.Dense(5, bias_initializer = bias_initializer, kernel_regularizer = kernel_regularizer)(inputs[:, (i * num_joint_data):((i + 1) * num_joint_data)])
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        x = tf.keras.layers.Dense(5, bias_initializer = bias_initializer, kernel_regularizer = kernel_regularizer)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        x = tf.keras.layers.Dense(5, bias_initializer = bias_initializer, kernel_regularizer = kernel_regularizer)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        x = tf.keras.layers.Dense(1, bias_initializer = bias_initializer, kernel_regularizer = kernel_regularizer)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        module_outputs.append(x)

    x = tf.keras.layers.Dense(5, bias_initializer = bias_initializer, kernel_regularizer = kernel_regularizer)(inputs[:, (num_joint * num_joint_data):])
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Dense(5, bias_initializer = bias_initializer, kernel_regularizer = kernel_regularizer)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Dense(1, bias_initializer = bias_initializer, kernel_regularizer = kernel_regularizer)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    module_outputs.append(x)

    concat_input = tf.keras.layers.concatenate(module_outputs)
    x = tf.keras.layers.Dense(5, bias_initializer = bias_initializer, kernel_regularizer = kernel_regularizer)(concat_input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Dense(5, bias_initializer = bias_initializer, kernel_regularizer = kernel_regularizer)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(2, activation = 'softmax', bias_initializer = bias_initializer, kernel_regularizer = kernel_regularizer)(x)
    model = tf.keras.Model(inputs = inputs, outputs = outputs)

    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate),
        loss = 'categorical_crossentropy',
        metrics = ['acc']
    )

    return model


if __name__ == "__main__":
    model = ModularizedNet()
    tf.keras.utils.plot_model(model, 'model.png', show_shapes = True)