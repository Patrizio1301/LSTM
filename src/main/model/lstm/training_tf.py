import tensorflow as tf


def train_network(model, num_epochs, config, num_steps=200, batch_size=32, state_size=4, verbose=True, save=False):
    feature = tf.compat.v1.placeholder(tf.int32, [batch_size, num_steps])
    target = tf.compat.v1.placeholder(tf.int32, [batch_size, num_steps])
    init_state = tf.zeros([batch_size, state_size])
    with tf.variable_scope('regression'):
        weights = tf.get_variable('W', [state_size, config.num_classes])
        bias = tf.get_variable('b', [config.num_classes], initializer=tf.constant_initializer(0.0))
    print(weights, bias)
    model = model(feature=feature,
                  target=target,
                  init_state=init_state,
                  weights=weights,
                  bias=bias,
                  config=config)
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        training_losses = []
        for idx, epoch in enumerate(gen_epochs(num_epochs, num_steps, batch_size)):
            training_loss = 0
            steps = 0
            training_state = None
            for X, Y in epoch:
                steps += 1
                feed_dict = {feature: X, target: Y, init_state: training_state}
                if training_state is not None:
                    feed_dict[g['init_state']] = training_state
                training_loss_, training_state, _ = sess.run([model.cost,
                                                              model.final_state,
                                                              model.optimization],
                                                             feed_dict)
                training_loss += training_loss_
            if verbose:
                print("Average training loss for Epoch", idx, ":", training_loss/steps)
            training_losses.append(training_loss/steps)

        # if isinstance(save, str):
        #     g['saver'].save(sess, save)
    return training_losses