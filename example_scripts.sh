#use a finetuned sst5 roberta as backbone
python sst5_proto.py -model_name="Unso/roberta-large-finetuned-sst5" -max_seq_length=50 -num_prototypes=1000 -hidden_shape=1024 -num_classes=5 -cohsep_ratio=0.5 -lr=6e-3 -proto_training_weights=1 -batch_size=64 -logger_dir="sst5_logs/setting1_logs" -checkpoint_dir="sst5_logs/setting1_logs" -config_subdir="0001" -max_epochs=10 -num_gpu=4
