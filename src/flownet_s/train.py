from ..dataloader import load_batch
from ..dataset_configs import FLYING_CHAIRS_DATASET_CONFIG
from ..training_schedules import LONG_SCHEDULE
from .flownet_s import FlowNetS
from ..flowlib import flow_to_image, write_flow

import tensorflow as tf
from ..deployment import model_deploy
slim = tf.contrib.slim

tf.app.flags.DEFINE_integer('num_clones', 1,
                            'Number of model clones to deploy.')

tf.app.flags.DEFINE_boolean('clone_on_cpu', False,
                            'Use CPUs to deploy clones.')

tf.app.flags.DEFINE_integer('worker_replicas', 1, 'Number of worker replicas.')

tf.app.flags.DEFINE_integer(
    'num_ps_tasks', 0,
    'The number of parameter servers. If the value is 0, then the parameters '
    'are handled locally by the worker.')

tf.app.flags.DEFINE_integer(
    'task', 0, 'Task id of the replica running the training.')

tf.app.flags.DEFINE_bool(
    'sync_replicas', False,
    'Whether or not to synchronize the replicas during training.')

tf.app.flags.DEFINE_integer(
    'replicas_to_aggregate', 1,
    'The Number of gradients to collect before updating params.')

FLAGS = tf.app.flags.FLAGS

with tf.Graph().as_default():
    #######################
    # Config model_deploy #
    #######################
    deploy_config = model_deploy.DeploymentConfig(
        num_clones=FLAGS.num_clones,
        clone_on_cpu=FLAGS.clone_on_cpu,
        replica_id=FLAGS.task,
        num_replicas=FLAGS.worker_replicas,
        num_ps_tasks=FLAGS.num_ps_tasks)

    # Create global_step
    #with tf.device(deploy_config.variables_device()):
    #  global_step = slim.create_global_step()

    # Create a new network
    net = FlowNetS()
    training_schedule = LONG_SCHEDULE
    log_dir='./logs/flownet_s_sample'

    # Load a batch of data
    input_a, input_b, flow = load_batch(FLYING_CHAIRS_DATASET_CONFIG, 'sample', net.global_step)

    batch_queue = slim.prefetch_queue.prefetch_queue(
                    [input_a, input_b, flow], capacity=2 * deploy_config.num_clones)

    # Define the optimizer
    net.learning_rate = tf.train.piecewise_constant(
        net.global_step,
        [tf.cast(v, tf.int64) for v in training_schedule['step_values']],
        training_schedule['learning_rates'])

    optimizer = tf.train.AdamOptimizer(
        net.learning_rate,
        training_schedule['momentum'],
        training_schedule['momentum2'])

    # Define the model
    def model_fn(batch_queue):
        input_a, input_b, flow = batch_queue.dequeue()            
        inputs = {
            'input_a': input_a,
            'input_b': input_b,
        }

        predictions = net.model(inputs, training_schedule=training_schedule)
        total_loss = net.loss(flow, predictions)
        tf.summary.scalar('loss', total_loss)

        if 'flow' in predictions:
            pred_flow_0 = predictions['flow'][0, :, :, :]
            pred_flow_0 = tf.py_func(flow_to_image, [pred_flow_0], tf.uint8)
            pred_flow_1 = predictions['flow'][1, :, :, :]
            pred_flow_1 = tf.py_func(flow_to_image, [pred_flow_1], tf.uint8)
            pred_flow_img = tf.stack([pred_flow_0, pred_flow_1], 0)
            tf.summary.image('pred_flow', pred_flow_img, max_outputs=2)

        true_flow_0 = flow[0, :, :, :]
        true_flow_0 = tf.py_func(flow_to_image, [true_flow_0], tf.uint8)
        true_flow_1 = flow[1, :, :, :]
        true_flow_1 = tf.py_func(flow_to_image, [true_flow_1], tf.uint8)
        true_flow_img = tf.stack([true_flow_0, true_flow_1], 0)
        tf.summary.image('true_flow', true_flow_img, max_outputs=2)

    model_dp = model_deploy.deploy(deploy_config, model_fn, [batch_queue], optimizer=optimizer)

    slim.learning.train(model_dp.train_op, 
                        log_dir,
                        session_config=tf.ConfigProto(allow_soft_placement=True),
                        number_of_steps=training_schedule['max_iter'])
    # Train on the data
    #net.train(
    #    log_dir='./logs/flownet_s_sample',
    #    training_schedule=LONG_SCHEDULE,
    #    batch_queue=batch_queue
    #)
