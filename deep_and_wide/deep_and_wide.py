
import os

import math
import paddle
import paddle.fluid as fluid
import sys


def train(use_cuda=False, is_sparse=True, is_local=True):
    PASS_NUM = 1000
    EMBED_SIZE = 1
    BATCH_SIZE = 32
    IS_SPARSE = is_sparse
    id_distributed = True

    dict_size = 10000

    word = fluid.layers.data(name='firstw', shape=[1], dtype='int64', lod_level=1)
    embed = fluid.layers.embedding(
        input=word,
        size=[dict_size, EMBED_SIZE],
        dtype='float32',
        is_sparse=IS_SPARSE,
        param_attr='shared_w',
        is_distributed=id_distributed)

    pool = fluid.layers.sequence_pool(input=embed, pool_type="sum")
    fc = fluid.layers.fc(input=pool, size=2, act="softmax")

    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    cost = fluid.layers.cross_entropy(input=fc, label=label)
    accuracy = fluid.layers.accuracy(input=fc, label=label)
    avg_cost = fluid.layers.mean(cost)

    sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.001)
    optimize_ops, params_grads = sgd_optimizer.minimize(avg_cost)

    word_dict = paddle.dataset.imdb.word_dict()
    train_data = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.imdb.train(word_dict), buf_size=1000),
        batch_size=BATCH_SIZE)
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)
    feeder = fluid.DataFeeder(feed_list=[word, label], place=place)

    def train_loop(main_program):
        exe.run(fluid.default_startup_program())

        for pass_id in xrange(PASS_NUM):
            for data in train_data():
                cost_val, acc_val = exe.run(main_program,
                                            feed=feeder.feed(data),
                                            fetch_list=[avg_cost, accuracy])
                print("cost=" + str(cost_val) + " acc=" + str(acc_val))
                if math.isnan(float(cost_val)):
                    sys.exit("got NaN loss, training failed.")

    if is_local:
        train_loop(fluid.default_main_program())
    else:
        port = os.getenv("PADDLE_INIT_PORT", "6174")
        pserver_ips = os.getenv("PADDLE_INIT_PSERVERS")  # ip,ip...
        eplist = []
        for ip in pserver_ips.split(","):
            eplist.append(':'.join([ip, port]))
        pserver_endpoints = ",".join(eplist)  # ip:port,ip:port...
        trainers = int(os.getenv("TRAINERS"))
        current_endpoint = os.getenv("POD_IP") + ":" + port
        trainer_id = int(os.getenv("PADDLE_INIT_TRAINER_ID"))
        training_role = os.getenv("TRAINING_ROLE", "TRAINER")
        t = fluid.DistributeTranspiler()
        t.transpile(trainer_id, pservers=pserver_endpoints, trainers=trainers)
        if training_role == "PSERVER":
            pserver_prog = t.get_pserver_program(current_endpoint)
            pserver_startup = t.get_startup_program(current_endpoint,
                                                    pserver_prog)
            exe.run(pserver_startup)
            exe.run(pserver_prog)
        elif training_role == "TRAINER":
            train_loop(t.get_trainer_program())

train()