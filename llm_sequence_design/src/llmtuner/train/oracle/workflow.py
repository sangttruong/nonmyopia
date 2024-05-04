from typing import TYPE_CHECKING, List, Optional

from ...data import get_dataset, split_dataset
from ...extras.callbacks import FixValueHeadModelCallback
from ...extras.misc import fix_valuehead_checkpoint
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from ..utils import create_modelcard_and_push
from .metric import compute_regression_metrics
from .trainer import OracleTrainer

import os
import json
import joblib
import numpy as np
from transformers import DefaultDataCollator
from sklearn.linear_model import LinearRegression, Ridge, BayesianRidge

if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, ModelArguments


def run_oracle(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    assert data_args.emb_enabled, "Oracle model only supports embedding enabled dataset"

    if not os.path.exists(model_args.model_name_or_path):
        if model_args.model_name_or_path.lower() == "linear":
            model = LinearRegression()
        elif model_args.model_name_or_path.lower() == "ridge":
            model = Ridge(alpha=10.0)
        elif model_args.model_name_or_path.lower() == "bayesridge":
            model = BayesianRidge()
        else:
            raise ValueError(
                f"model_name_or_path {model_args.model_name_or_path} is not supported"
            )
    else:
        model = joblib.load(os.path.join(
            model_args.model_name_or_path, 'model.joblib'))

    # Training
    if training_args.do_train:
        print("Training oracle model...")
        data_args.split = "train"
        train_dataset = get_dataset(None, model_args,
                                    data_args, training_args, stage="oracle")

        X_train = train_dataset.data["inputs_embeds"].to_numpy()
        y_train = train_dataset.data["rewards"].to_numpy()
        X_train = np.stack(X_train)
        y_train = np.stack(y_train)
        model.fit(X_train, y_train)

        # Save model
        os.makedirs(training_args.output_dir, exist_ok=True)
        joblib.dump(model, os.path.join(
            training_args.output_dir, 'model.joblib'))

        # Save results
        y_train_hat = model.predict(X_train)
        train_metrics = compute_regression_metrics((y_train_hat, y_train))
        train_metrics = {f"train_{k}": v for k, v in train_metrics.items()}
        with open(os.path.join(training_args.output_dir, 'train_results.json'), 'w') as f:
            json.dump(train_metrics, f)

    # Evaluation
    if training_args.do_eval:
        print("Evaluating oracle model...")
        data_args.split = "validation"
        eval_dataset = get_dataset(None, model_args,
                                   data_args, training_args, stage="oracle")

        X_test = eval_dataset.data["inputs_embeds"].to_numpy()
        y_test = eval_dataset.data["rewards"].to_numpy()
        X_test = np.stack(X_test)
        y_test = np.stack(y_test)

        # Save results
        y_test_hat = model.predict(X_test)
        eval_metrics = compute_regression_metrics((y_test_hat, y_test))
        eval_metrics = {f"eval_{k}": v for k, v in eval_metrics.items()}
        os.makedirs(training_args.output_dir, exist_ok=True)
        with open(os.path.join(training_args.output_dir, 'eval_results.json'), 'w') as f:
            json.dump(eval_metrics, f)

    # Predict
    if training_args.do_predict:
        print("Predicting oracle model...")
        data_args.split = "validation"
        eval_dataset = get_dataset(None, model_args,
                                   data_args, training_args, stage="oracle")

        X_test = eval_dataset.data["inputs_embeds"].to_numpy()
        y_test = eval_dataset.data["rewards"].to_numpy()
        X_test = np.stack(X_test)
        y_test = np.stack(y_test)

        # Save to jsonl file
        y_test_hat = model.predict(X_test)
        with open(os.path.join(training_args.output_dir, 'predictions.jsonl'), 'w') as f:
            for i in range(len(y_test)):
                json.dump({"inputs_embeds": X_test[i].tolist(
                ), "rewards": y_test[i], "rewards_hat": y_test_hat[i]}, f)


def run_oracle_pytorch(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    tokenizer = load_tokenizer(model_args)
    data_args.split = "train"
    train_dataset = get_dataset(tokenizer, model_args,
                                data_args, training_args, stage="oracle")
    data_args.split = "validation"
    eval_dataset = get_dataset(tokenizer, model_args,
                               data_args, training_args, stage="oracle")
    model = load_model(tokenizer, model_args, finetuning_args,
                       training_args.do_train, add_valuehead=True, emb_enabled=data_args.emb_enabled)

    # Update arguments
    training_args.remove_unused_columns = False  # important for pairwise dataset
    if data_args.emb_enabled:
        data_collator = DefaultDataCollator()
    else:
        data_collator = None

    # Initialize our Trainer
    trainer = OracleTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        tokenizer=tokenizer,
        callbacks=callbacks + [FixValueHeadModelCallback()],
        compute_metrics=compute_regression_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        emb_enabled=data_args.emb_enabled,
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train(
            resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        if training_args.should_save:
            fix_valuehead_checkpoint(
                model, training_args.output_dir, training_args.save_safetensors)
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            plot_loss(training_args.output_dir, keys=[
                      "loss", "eval_loss", "eval_rmse"])

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        predict_results = trainer.predict(
            eval_dataset, metric_key_prefix="predict")
        trainer.log_metrics("predict", predict_results.metrics)
        trainer.save_metrics("predict", predict_results.metrics)
        trainer.save_predictions(predict_results)

    # Create model card
    create_modelcard_and_push(
        trainer, model_args, data_args, training_args, finetuning_args)
