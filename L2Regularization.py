import tensorflow as tf
import tools
import logging

def L2RegularizationLoss(args):
    print('')
    with tf.name_scope('RegLoss'):
        logging.info('L2 regularization loss.')
        vars_to_train = tools.get_trainable_variables(args)
        reg_loss = tf.zeros((), dtype=tf.float32, name='zeros')
        for var in vars_to_train:
            add_to_reg = True
            for string_not_add in args.vars_to_skip_l2_reg:
                if string_not_add in var.name:
                    add_to_reg = False
                    break
            if add_to_reg:
                logging.info('Adding ' + var.name + ' to regularization loss')
                reg_loss += tf.reduce_sum(tf.square(var, name='square'), name='reduce_sum')
            else:
                logging.info('Not adding ' + var.name + ' to regularization loss')
        # reg_loss *= args.l2_regularization / 2.0
        reg_loss = tf.multiply(reg_loss, args.l2_regularization, name='reg_loss')

    tf.summary.scalar('losses/reg_loss', reg_loss)
    return reg_loss