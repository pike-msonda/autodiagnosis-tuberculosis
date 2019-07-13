import numpy as np
import sys
sys.path.append("..") 
from utils.data_utils import *
import tensorflow as tf
from datetime import datetime
from tf_alexnet import AlexNet
from sklearn.model_selection import train_test_split
from image_datagenerator import ImageDataGenerator

DATASET_PATH = '../data/aug/all'
IMAGESET_NAME = os.path.join(DATASET_PATH, 'usa.pkl')

if __name__ == "__main__":
    # Learning params
    learning_rate = 0.01
    num_epochs = 40
    batch_size = 32
    display_step = 1
    dropout_rate = 0.5
    num_classes = 2
    train_layers = ['fc8', 'fc7']

    x = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
    y = tf.placeholder(tf.float32, [None, num_classes])
    keep_prob = tf.placeholder(tf.float32)

    filewriter_path = "/tmp/finetune_alexnet/tb"
    checkpoint_path = "/tmp/finetune_alexnet/"
    # Initialize model
    model = AlexNet(x, keep_prob, num_classes, train_layers)

    xdata, ydata = build_image_dataset_from_dir(os.path.join(DATASET_PATH, 'usa'),
    dataset_file=IMAGESET_NAME,
    resize=None,
    filetypes=['.png'],
    convert_to_color=True,
    shuffle_data=True,
    categorical_Y=True)

    X_train, X_test, y_train, y_test = train_test_split(xdata,ydata, test_size=0.30, 
        random_state=1000)
    #link variable to model output
    
    score = model.fc8
    print(score)

    # List of trainable variables of the layers we want to train
    var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]

    # Op for calculating the loss
    with tf.name_scope("cross_ent"):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = score, labels = y))

        # Train op
    with tf.name_scope("train"):
    # Get gradients of all trainable variables
        gradients = tf.gradients(loss, var_list)
        gradients = list(zip(gradients, var_list))

        # Create optimizer and apply gradient descent to the trainable variables
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_op = optimizer.apply_gradients(grads_and_vars=gradients)

    # Add gradients to summary
    for gradient, var in gradients:
        tf.summary.histogram(var.name + '/gradient', gradient)

    # Add the variables we train to the summary
    for var in var_list:
        tf.summary.histogram(var.name, var)

    # Add the loss to summary
    tf.summary.scalar('cross_entropy', loss)

    with tf.name_scope("accuracy"):
        correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # Add the accuracy to the summary
        tf.summary.scalar('accuracy', accuracy)
        
    # Merge all summaries together
    merged_summary = tf.summary.merge_all()

    # Initialize the FileWriter
    writer = tf.summary.FileWriter(filewriter_path)

    # Initialize an saver for store model checkpoints
    saver = tf.train.Saver()

    # Initalize the data generator seperately for the training and validation set
    train_generator = ImageDataGenerator(X_train, y_train,
                                        horizontal_flip = False, shuffle = True)
    val_generator = ImageDataGenerator(X_test,y_test, shuffle = False)
    # Get the number of training/validation steps per epoch
    # import pdb; pdb.set_trace()
    train_batches_per_epoch = np.floor(train_generator.data_size / batch_size).astype(np.int16)
    val_batches_per_epoch = np.floor(val_generator.data_size / batch_size).astype(np.int16)

    with tf.Session() as sess:

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        # Add the model graph to TensorBoard
        writer.add_graph(sess.graph)

        # Load the pretrained weights into the non-trainable layer
        model.load_initial_weights(sess)

        print("{} Start training...".format(datetime.now()))
        print("{} Open Tensorboard at --logdir {}".format(datetime.now(),
                                                            filewriter_path))

        # Loop over number of epochs
        for epoch in range(num_epochs):

                print("{} Epoch number: {}".format(datetime.now(), epoch+1))

                step = 1

                while step < train_batches_per_epoch:

                    # Get a batch of images and labels
                    batch_xs, batch_ys = train_generator.next_batch(batch_size)

                    # And run the training op
                    sess.run(train_op, feed_dict={x: batch_xs,
                                                y: batch_ys,
                                                keep_prob: dropout_rate})

                    # Generate summary with the current batch of data and write to file
                    if step%display_step == 0:
                        s = sess.run(merged_summary, feed_dict={x: batch_xs,
                                                                y: batch_ys,
                                                                keep_prob: 1.})
                        writer.add_summary(s, epoch*train_batches_per_epoch + step)

                    step += 1

                # Validate the model on the entire validation set
                print("{} Start validation".format(datetime.now()))
                test_acc = 0.
                test_count = 0
                # import pdb; pdb.set_trace()
                for _ in range(val_batches_per_epoch):
                    batch_tx, batch_ty = val_generator.next_batch(batch_size)
                    acc = sess.run(accuracy, feed_dict={x: batch_tx,
                                                        y: batch_ty,
                                                        keep_prob: 1.})
                    test_acc += acc
                    test_count += 1
                test_acc /= test_count
                print("Validation Accuracy = at {0} of {1}".format(datetime.now(), test_acc))

                # Reset the file pointer of the image data generator
                val_generator.reset_pointer()
                train_generator.reset_pointer()

                print("{} Saving checkpoint of model...".format(datetime.now()))

                #save checkpoint of the model
                checkpoint_name = os.path.join(checkpoint_path, 'model_epoch'+str(epoch)+'.ckpt')
                save_path = saver.save(sess, checkpoint_name)

                print("{} Model checkpoint saved at {}".format(datetime.now(), checkpoint_name))