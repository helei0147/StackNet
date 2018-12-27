import numpy as np
import tensorflow as tf
import time
from datetime import datetime
from scipy import misc
import os

import deepReflMaps

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('max_steps', 30000,
                            """Number of batchs to run.""")


def train():
    with tf.Graph().as_default():

        # 全局计数器
        global_step = tf.Variable(0, trainable=False)

        # 读入图片
        images, labels = deepReflMaps.inputs(False)

        # 计算 ReflectMaps
        logits = deepReflMaps.inference(images, train=True, AngFlag=False)

        # 计算损失函数
        loss = deepReflMaps.loss(logits, labels)
        train_op = deepReflMaps.training(loss, global_step)

        saver = tf.train.Saver(tf.global_variables())

        init = tf.global_variables_initializer()
        sess = tf.Session()

        sess.run(init)

        tf.train.start_queue_runners(sess=sess)

        for step in range(FLAGS.max_steps):
            start_time = time.time()
            _, loss_value, logit, label = sess.run([train_op, loss, logits, labels])
            # tf.summary.histogram(var.op.name + '/loss', loss)
            # tf.summary.histogram(var.op.name + '/labels', labels)

            duration = time.time() - start_time

            try:
                assert not np.isnan(loss_value)
            except:
                raise ValueError('Model diverged with loss = NaN in step %d' % step)

            if step % 10 == 0:
                num_examples_per_step = FLAGS.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = (
                    '%s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)')
                print(format_str % (datetime.now(), step, loss_value,
                                    examples_per_sec, sec_per_batch))

            # if step % 100 == 0:
            #     summary_str = sess.run(summary_op)
            #     summary_writer.add_summary(summary_str, step)
            if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)


    sess.close()

def main(argv=None):
    train()

if __name__ == '__main__':
    tf.app.run()
