Repository for **A Novel Prototypical Network-Based Method for Built-in Interpretability
in Large Language Models**

The environment file is included. For reference, major dependencies are listed below:

```
pytorch=1.11 
transformers==4.29.2
pytorch-lightning==1.7.7
torchcluster==0.1.4
torchmetrics==0.11.0
tensorboard==2.10.1
```

For each NLP task the code is provided in {TASK_NAME}_proto.py <br>
Example script: 
```
python sst5_proto.py -model_name="Unso/roberta-large-finetuned-sst5" -max_seq_length=50 -num_prototypes=1000 -hidden_shape=1024 -num_classes=5 -cohsep_ratio=0.5 -lr=6e-3 -proto_training_weights=1 -batch_size=64 -logger_dir="sst5_logs/setting1_logs" -checkpoint_dir="sst5_logs/setting1_logs" -config_subdir="0001" -max_epochs=10 -num_gpu=4
```


