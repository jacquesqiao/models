import deep_and_wide

deep_and_wide.train(
    is_local=False,
    training_role="TRAINER",
    pserver_endpoints="127.0.0.1:6000,127.0.0.1:6001",       # ip:port,ip:port list
    current_endpoint="127.0.0.1:6000",    # ip:port the ip port for this pserver
    trainer_num=1,
    trainer_id=0,
)
