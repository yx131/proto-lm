import argparse

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification


from ProtoLM import proto_lm
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

import datasets


class sst_datamodule(pl.LightningDataModule):
    loader_columns = [
        "datasets_idx",
        "input_ids",
        "token_type_ids",
        "attention_mask",
        "start_positions",
        "end_positions",
        "labels",
    ]

    def __init__(
            self,
            model_name_or_path: str,
            max_seq_length: int = 128,
            train_batch_size: int = 32,
            eval_batch_size: int = 32,
            **kwargs,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

        self.text_fields = ['text']
        self.num_labels = 5
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def setup(self, stage: str):
        self.dataset = datasets.load_dataset("SetFit/sst5")

        for split in self.dataset.keys():
            print(f'split is: {split}')
            self.dataset[split] = self.dataset[split].map(
                self.convert_to_features,
                batched=True,
                remove_columns=["label"],
            )
            self.columns = [c for c in self.dataset[split].column_names if c in self.loader_columns]
            print(f'self.columns: {self.columns}')
            self.dataset[split].set_format(type="torch", columns=self.columns)

        self.eval_splits = [x for x in self.dataset.keys() if "validation" in x]


    def train_dataloader(self):
        print(f'returned self batch size: {self.train_batch_size}')
        return DataLoader(self.dataset["train"], batch_size=self.train_batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dataset["test"], batch_size=self.eval_batch_size)

    def test_dataloader(self):
        return DataLoader(self.dataset["test"], batch_size=self.eval_batch_size)
    
    
    def convert_to_features(self, example_batch, indices=None):

        if len(self.text_fields) > 1:
            texts_or_text_pairs = list(zip(example_batch[self.text_fields[0]], example_batch[self.text_fields[1]]))
        else:
            texts_or_text_pairs = example_batch[self.text_fields[0]]

        # modified tokenizer
        features = self.tokenizer.batch_encode_plus(
            texts_or_text_pairs,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.max_seq_length
        )

        # Rename label to labels to make it easier to pass to model forward
        features["labels"] = example_batch["label"]

        return features

    
parser = argparse.ArgumentParser()
parser.add_argument('-model_name', default=None, type=str, help='backbone LLM model ot load')
parser.add_argument('-max_seq_length', default=50, type=int, help='maximum sentence length to pad/truncate to')
parser.add_argument('-num_prototypes', default=200, type=int, help='number of prototypes to train')
parser.add_argument('-hidden_shape', default=1024, type=int, help='hidden shape of each prototype, should be same as output hidden shape of underlying llm')
parser.add_argument('-num_classes', default=2, type=int, help='number of output classes')
parser.add_argument('-cohsep_ratio', default=0.5, type=float, help='ratio of prototypes in class to push/pull')
parser.add_argument('-lambda0', default=0.5, type=float, help='lambda0 in loss')
parser.add_argument('-lambda1', default=0.25, type=float, help='lambda1 in loss')
parser.add_argument('-lambda2', default=0.25, type=float, help='lambda2 in loss')
parser.add_argument('-lr', default=3e-4, type=float, help='initial learning rate')
parser.add_argument('-proto_training_weights', default=1, type=int, help='number of prototypes to train')
parser.add_argument('-batch_size', default=128, type=int, help='batch size for dataloader')
parser.add_argument('-logger_dir', default='tb_logs', type=str, help='directory for the logger to store training details')
parser.add_argument('-checkpoint_dir', default='ckpt_dir', type=str, help='directory to store checkpoints')
parser.add_argument('-config_subdir', default='config_subdir', type=str, help='subdirectory for checkpoints of a certain config')
parser.add_argument('-max_epochs', default=2, type=int, help='number of epochs to train')
parser.add_argument('-num_gpu', default=2, type=int, help='number of gpus to train on')
parser.add_argument('-load_model', default='', type=str, help='to load a pretrained model, if there is one')

if __name__ == '__main__':
    args = parser.parse_args()
    print(f'args: {args}')

    # get data module
    sst5_dm = sst_datamodule(model_name_or_path=args.model_name,
                        max_seq_length=args.max_seq_length,
                        train_batch_size=args.batch_size,
                        eval_batch_size=args.batch_size)


    sst5_dm.setup(stage='fit')

    # if loading a trained model
    if args.load_model != '':
        print(f'loading a model: {args.load_model}')
        proto = proto_lm.load_from_checkpoint(args.load_model, pretrained_model= AutoModelForSequenceClassification.from_pretrained(args.model_name, ignore_mismatched_sizes=True).roberta)
    else:
        # otherwise build proto object

        # config = AutoConfig.from_pretrained(args.model_name)
        llm_model = AutoModelForSequenceClassification.from_pretrained(args.model_name, ignore_mismatched_sizes=True).roberta #roberta based model

        proto = proto_lm(pretrained_model=llm_model,
                          max_seq_length=args.max_seq_length,
                          num_prototypes=args.num_prototypes,
                          hidden_shape=args.hidden_shape,
                          num_classes=args.num_classes,
                          cohsep_ratio=args.cohsep_ratio,
                          lambda0=args.lambda0,
			  lambda1=args.lambda1,
                          lambda2=args.lambda2,
                          lr=args.lr,
                          proto_training_weights=bool(args.proto_training_weights),
                          )

    # get training utilities like logger and checckpoints
    tb_logger = TensorBoardLogger(f'{args.logger_dir}', name=f'sst5_tensorboard_logs')
    ckpt_path = f'{args.checkpoint_dir}/{args.config_subdir})'
    checkpoint_callback = ModelCheckpoint(dirpath=f'{ckpt_path}',
                                          monitor='val_loss',
                                          save_top_k=3,
                                          filename="{epoch}-{val_loss:.4f}-{val_accuracy:.4f}"
                                          )

    # get trainer object
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices=args.num_gpu if torch.cuda.is_available() else None,
        track_grad_norm=1,
        strategy='ddp',
        logger=tb_logger,
        callbacks=[checkpoint_callback]
    )

    trainer.fit(proto, datamodule=sst5_dm)

    #trainer.test(proto, datamodule=sst5_dm, ckpt_path=args.load_model)