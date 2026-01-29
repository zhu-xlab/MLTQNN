import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
import wandb
import os
import shutil
import sys
from itertools import chain
import operator

class ChangeLossWeightCallback(tf.keras.callbacks.Callback):
    def __init__(self, warmup, alpha, beta, weight_ratio):
        super(ChangeLossWeightCallback, self).__init__()
        self.warmup = warmup
        self.alpha = alpha
        self.beta = beta
        self.weight_ratio = weight_ratio
        
    def on_epoch_end(self, epoch, logs=None):
        if epoch >= self.warmup:
            K.set_value(self.alpha, 1.0)
            K.set_value(self.beta, self.weight_ratio*self.alpha)            
class EarlyStopping(tf.keras.callbacks.Callback):
    def __init__(self, monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto', baseline=None, restore_best_weights=False, warmup = 0):
        super(EarlyStopping, self).__init__()
        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.baseline = baseline
        self.min_delta = abs(min_delta)
        self.wait = 0
        self.stopped_epoch = 0
        self.restore_best_weights = restore_best_weights
        self.best_weights = None
        self.warmup = warmup

        if mode not in ['auto', 'min', 'max']:
            mode = 'auto'
        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            if (self.monitor.endswith('acc') or self.monitor.endswith('accuracy') or self.monitor.endswith('auc')):
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less
        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1
    
    def on_train_begin(self, logs=None):
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf
        self.best_weights = None
        self.best_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        if epoch > self.warmup:
            current = self.get_monitor_value(logs)
            if current is None:
                return
            if self.restore_best_weights and self.best_weights is None:
                self.best_weights = self.model.get_weights()
                
            self.wait += 1
            if self._is_improvement(current, self.best):
                self.best = current
                self.best_epoch = epoch
                if self.restore_best_weights:
                    self.best_weights = self.model.get_weights()
                if self.baseline is None or self._is_improvement(current, self.baseline):
                    self.wait = 0

            if self.wait >= self.patience and epoch > 0:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                if self.restore_best_weights and self.best_weights is not None:
                    if self.verbose > 0:
                        print('Restoring model weights from the end of the best epoch: 'f'{self.best_epoch + 1}.')
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))

    def get_monitor_value(self, logs):
        logs = logs or {}
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            logging.warning('Early stopping conditioned on metric `%s` ''which is not available. Available metrics are: %s',self.monitor, ','.join(list(logs.keys())))
        return monitor_value

    def _is_improvement(self, monitor_value, reference_value):
        return self.monitor_op(monitor_value - self.min_delta, reference_value)
    
    
class WandbCallback(tf.keras.callbacks.Callback):
    def __init__(
        self,
        monitor="val_loss",
        verbose=0,
        mode="auto",
        save_weights_only=False,
        log_weights=False,
        log_gradients=False,
        save_model=True,
        training_data=None,
        validation_data=None,
        labels=[],
        predictions=36,
        generator=None,
        input_type=None,
        output_type=None,
        log_evaluation=False,
        validation_steps=None,
        class_colors=None,
        log_batch_frequency=None,
        log_best_prefix="best_",
        save_graph=True,
        validation_indexes=None,
        validation_row_processor=None,
        prediction_row_processor=None,
        infer_missing_processors=True,
        log_evaluation_frequency=0,
        warmup=0,
        **kwargs,
    ):
        if wandb.run is None:
            raise wandb.Error("You must call wandb.init() before WandbCallback()")
        with wandb.wandb_lib.telemetry.context(run=wandb.run) as tel:
            tel.feature.keras = True
        self.validation_data = None
        # This is kept around for legacy reasons
        if validation_data is not None:
            if is_generator_like(validation_data):
                generator = validation_data
            else:
                self.validation_data = validation_data

        self.labels = labels
        self.predictions = min(predictions, 100)

        self.monitor = monitor
        self.verbose = verbose
        self.save_weights_only = save_weights_only
        self.save_graph = save_graph

        wandb.save("model-best.h5")
        self.filepath = os.path.join(wandb.run.dir, "model-best.h5")
        self.save_model = save_model
        # if save_model:
        #     deprecate(
        #         field_name=Deprecated.keras_callback__save_model,
        #         warning_message=(
        #             "The save_model argument by default saves the model in the HDF5 format that cannot save "
        #             "custom objects like subclassed models and custom layers. This behavior will be deprecated "
        #             "in a future release in favor of the SavedModel format. Meanwhile, the HDF5 model is saved "
        #             "as W&B files and the SavedModel as W&B Artifacts."
        #         ),
        #     )

        self.save_model_as_artifact = False
        self.log_weights = log_weights
        self.log_gradients = log_gradients
        self.training_data = training_data
        self.generator = generator
        self._graph_rendered = False

        data_type = kwargs.get("data_type", None)
        if data_type is not None:
            # deprecate(
            #     field_name=Deprecated.keras_callback__data_type,
            #     warning_message=(
            #         "The data_type argument of wandb.keras.WandbCallback is deprecated "
            #         "and will be removed in a future release. Please use input_type instead.\n"
            #         "Setting input_type = data_type."
            #     ),
            # )
            input_type = data_type
        self.input_type = input_type
        self.output_type = output_type
        self.log_evaluation = log_evaluation
        self.validation_steps = validation_steps
        self.class_colors = np.array(class_colors) if class_colors is not None else None
        self.log_batch_frequency = log_batch_frequency
        self.log_best_prefix = log_best_prefix

        self._prediction_batch_size = None

        if self.log_gradients:
            if int(tf.__version__.split(".")[0]) < 2:
                raise Exception("Gradient logging requires tensorflow 2.0 or higher.")
            if self.training_data is None:
                raise ValueError(
                    "training_data argument is required for gradient logging."
                )
            if isinstance(self.training_data, (list, tuple)):
                if len(self.training_data) != 2:
                    raise ValueError("training data must be a tuple of length two")
                self._training_data_x, self._training_data_y = self.training_data
            else:
                self._training_data_x = (
                    self.training_data
                )  # generator, tf.data.Dataset etc
                self._training_data_y = None

        # From Keras
        if mode not in ["auto", "min", "max"]:
            print(f"WandbCallback mode {mode} is unknown, fallback to auto mode.")
            mode = "auto"

        if mode == "min":
            self.monitor_op = operator.lt
            self.best = float("inf")
        elif mode == "max":
            self.monitor_op = operator.gt
            self.best = float("-inf")
        else:
            if "acc" in self.monitor or self.monitor.startswith("fmeasure"):
                self.monitor_op = operator.gt
                self.best = float("-inf")
            else:
                self.monitor_op = operator.lt
                self.best = float("inf")
        # Get the previous best metric for resumed runs
        previous_best = wandb.run.summary.get(f"{self.log_best_prefix}{self.monitor}")
        if previous_best is not None:
            self.best = previous_best

        self._validation_data_logger = None
        self._validation_indexes = validation_indexes
        self._validation_row_processor = validation_row_processor
        self._prediction_row_processor = prediction_row_processor
        self._infer_missing_processors = infer_missing_processors
        self._log_evaluation_frequency = log_evaluation_frequency
        self._model_trained_since_last_eval = False
        
        self.warmup = warmup

    def _build_grad_accumulator_model(self):
        inputs = self.model.inputs
        outputs = self.model(inputs)
        grad_acc_model = tf.keras.models.Model(inputs, outputs)
        grad_acc_model.compile(loss=self.model.loss, optimizer=_CustomOptimizer())

        # make sure magic doesn't think this is a user model
        grad_acc_model._wandb_internal_model = True

        self._grad_accumulator_model = grad_acc_model
        self._grad_accumulator_callback = _GradAccumulatorCallback()

    def _implements_train_batch_hooks(self):
        return self.log_batch_frequency is not None

    def _implements_test_batch_hooks(self):
        return self.log_batch_frequency is not None

    def _implements_predict_batch_hooks(self):
        return self.log_batch_frequency is not None

    def set_params(self, params):
        self.params = params

    def set_model(self, model):
        self.model = model
        if self.input_type == "auto" and len(model.inputs) == 1:
            self.input_type = wandb.util.guess_data_type(
                model.inputs[0].shape, risky=True
            )
        if self.input_type and self.output_type is None and len(model.outputs) == 1:
            self.output_type = wandb.util.guess_data_type(model.outputs[0].shape)
        if self.log_gradients:
            self._build_grad_accumulator_model()

    def _attempt_evaluation_log(self, commit=True):
        if self.log_evaluation and self._validation_data_logger:
            try:
                if not self.model:
                    wandb.termwarn("WandbCallback unable to read model from trainer")
                else:
                    self._validation_data_logger.log_predictions(
                        predictions=self._validation_data_logger.make_predictions(
                            self.model.predict
                        ),
                        commit=commit,
                    )
                    self._model_trained_since_last_eval = False
            except Exception as e:
                wandb.termwarn("Error durring prediction logging for epoch: " + str(e))

    def on_epoch_end(self, epoch, logs={}):
        if self.log_weights:
            wandb.log(self._log_weights(), commit=False)

        if self.log_gradients:
            wandb.log(self._log_gradients(), commit=False)

        if self.input_type in (
            "image",
            "images",
            "segmentation_mask",
        ) or self.output_type in ("image", "images", "segmentation_mask"):
            if self.generator:
                self.validation_data = next(self.generator)
            if self.validation_data is None:
                wandb.termwarn(
                    "No validation_data set, pass a generator to the callback."
                )
            elif self.validation_data and len(self.validation_data) > 0:
                wandb.log(
                    {"examples": self._log_images(num_images=self.predictions)},
                    commit=False,
                )

        if (
            self._log_evaluation_frequency > 0
            and epoch % self._log_evaluation_frequency == 0
        ):
            self._attempt_evaluation_log(commit=False)

        wandb.log({"epoch": epoch}, commit=False)
        wandb.log(logs, commit=True)

        self.current = logs.get(self.monitor)
        if epoch > self.warmup:
            if self.current and self.monitor_op(self.current, self.best):
                if self.log_best_prefix:
                    wandb.run.summary[
                        f"{self.log_best_prefix}{self.monitor}"
                    ] = self.current
                    wandb.run.summary["{}{}".format(self.log_best_prefix, "epoch")] = epoch
                    if self.verbose and not self.save_model:
                        print(
                            "Epoch %05d: %s improved from %0.5f to %0.5f"
                            % (epoch, self.monitor, self.best, self.current)
                        )
                if self.save_model:
                    self._save_model(epoch)

                if self.save_model_as_artifact:
                    self._save_model_as_artifact(epoch)

                self.best = self.current

    # This is what keras used pre tensorflow.keras
    def on_batch_begin(self, batch, logs=None):
        pass

    # This is what keras used pre tensorflow.keras
    def on_batch_end(self, batch, logs=None):
        if self.save_graph and not self._graph_rendered:
            # Couldn't do this in train_begin because keras may still not be built
            wandb.run.summary["graph"] = wandb.Graph.from_keras(self.model)
            self._graph_rendered = True

        if self.log_batch_frequency and batch % self.log_batch_frequency == 0:
            wandb.log(logs, commit=True)

    def on_train_batch_begin(self, batch, logs=None):
        self._model_trained_since_last_eval = True

    def on_train_batch_end(self, batch, logs=None):
        if self.save_graph and not self._graph_rendered:
            # Couldn't do this in train_begin because keras may still not be built
            wandb.run.summary["graph"] = wandb.Graph.from_keras(self.model)
            self._graph_rendered = True

        if self.log_batch_frequency and batch % self.log_batch_frequency == 0:
            wandb.log(logs, commit=True)

    def on_test_begin(self, logs=None):
        pass

    def on_test_end(self, logs=None):
        pass

    def on_test_batch_begin(self, batch, logs=None):
        pass

    def on_test_batch_end(self, batch, logs=None):
        pass

    def on_train_begin(self, logs=None):
        if self.log_evaluation:
            try:
                validation_data = None
                if self.validation_data:
                    validation_data = self.validation_data
                elif self.generator:
                    if not self.validation_steps:
                        wandb.termwarn(
                            "WandbCallback is unable to log validation data. When using a generator for validation_data, you must pass validation_steps"
                        )
                    else:
                        x = None
                        y_true = None
                        for i in range(self.validation_steps):
                            bx, by_true = next(self.generator)
                            if x is None:
                                x, y_true = bx, by_true
                            else:
                                x, y_true = (
                                    np.append(x, bx, axis=0),
                                    np.append(y_true, by_true, axis=0),
                                )
                        validation_data = (x, y_true)
                else:
                    wandb.termwarn(
                        "WandbCallback is unable to read validation_data from trainer and therefore cannot log validation data. Ensure Keras is properly patched by calling `from wandb.keras import WandbCallback` at the top of your script."
                    )
                if validation_data:
                    self._validation_data_logger = ValidationDataLogger(
                        inputs=validation_data[0],
                        targets=validation_data[1],
                        indexes=self._validation_indexes,
                        validation_row_processor=self._validation_row_processor,
                        prediction_row_processor=self._prediction_row_processor,
                        class_labels=self.labels,
                        infer_missing_processors=self._infer_missing_processors,
                    )
            except Exception as e:
                wandb.termwarn(
                    "Error initializing ValidationDataLogger in WandbCallback. Skipping logging validation data. Error: "
                    + str(e)
                )

    def on_train_end(self, logs=None):
        if self._model_trained_since_last_eval:
            self._attempt_evaluation_log()

    def on_test_begin(self, logs=None):
        pass

    def on_test_end(self, logs=None):
        pass

    def on_test_batch_begin(self, batch, logs=None):
        pass

    def on_test_batch_end(self, batch, logs=None):
        pass

    def on_predict_begin(self, logs=None):
        pass

    def on_predict_end(self, logs=None):
        pass

    def on_predict_batch_begin(self, batch, logs=None):
        pass

    def on_predict_batch_end(self, batch, logs=None):
        pass

    def _logits_to_captions(self, logits):
        if logits[0].shape[-1] == 1:
            # Scalar output from the model
            # TODO: handle validation_y
            if len(self.labels) == 2:
                # User has named true and false
                captions = [
                    self.labels[1] if logits[0] > 0.5 else self.labels[0]
                    for logit in logits
                ]
            else:
                if len(self.labels) != 0:
                    wandb.termwarn(
                        'keras model is producing a single output, so labels should be a length two array: ["False label", "True label"].'
                    )
                captions = [logit[0] for logit in logits]
        else:
            # Vector output from the model
            # TODO: handle validation_y
            labels = np.argmax(np.stack(logits), axis=1)

            if len(self.labels) > 0:
                # User has named the categories in self.labels
                captions = []
                for label in labels:
                    try:
                        captions.append(self.labels[label])
                    except IndexError:
                        captions.append(label)
            else:
                captions = labels
        return captions

    def _masks_to_pixels(self, masks):
        # if its a binary mask, just return it as grayscale instead of picking the argmax
        if len(masks[0].shape) == 2 or masks[0].shape[-1] == 1:
            return masks
        class_colors = (
            self.class_colors
            if self.class_colors is not None
            else np.array(wandb.util.class_colors(masks[0].shape[2]))
        )
        imgs = class_colors[np.argmax(masks, axis=-1)]
        return imgs

    def _log_images(self, num_images=36):
        validation_X = self.validation_data[0]
        validation_y = self.validation_data[1]

        validation_length = len(validation_X)

        if validation_length > num_images:
            # pick some data at random
            indices = np.random.choice(validation_length, num_images, replace=False)
        else:
            indices = range(validation_length)

        test_data = []
        test_output = []
        for i in indices:
            test_example = validation_X[i]
            test_data.append(test_example)
            test_output.append(validation_y[i])

        if self.model.stateful:
            predictions = self.model.predict(np.stack(test_data), batch_size=1)
            self.model.reset_states()
        else:
            predictions = self.model.predict(
                np.stack(test_data), batch_size=self._prediction_batch_size
            )
            if len(predictions) != len(test_data):
                self._prediction_batch_size = 1
                predictions = self.model.predict(
                    np.stack(test_data), batch_size=self._prediction_batch_size
                )

        if self.input_type == "label":
            if self.output_type in ("image", "images", "segmentation_mask"):
                captions = self._logits_to_captions(test_data)
                output_image_data = (
                    self._masks_to_pixels(predictions)
                    if self.output_type == "segmentation_mask"
                    else predictions
                )
                reference_image_data = (
                    self._masks_to_pixels(test_output)
                    if self.output_type == "segmentation_mask"
                    else test_output
                )
                output_images = [
                    wandb.Image(data, caption=captions[i], grouping=2)
                    for i, data in enumerate(output_image_data)
                ]
                reference_images = [
                    wandb.Image(data, caption=captions[i])
                    for i, data in enumerate(reference_image_data)
                ]
                return list(chain.from_iterable(zip(output_images, reference_images)))
        elif self.input_type in ("image", "images", "segmentation_mask"):
            input_image_data = (
                self._masks_to_pixels(test_data)
                if self.input_type == "segmentation_mask"
                else test_data
            )
            if self.output_type == "label":
                # we just use the predicted label as the caption for now
                captions = self._logits_to_captions(predictions)
                return [
                    wandb.Image(data, caption=captions[i])
                    for i, data in enumerate(test_data)
                ]
            elif self.output_type in ("image", "images", "segmentation_mask"):
                output_image_data = (
                    self._masks_to_pixels(predictions)
                    if self.output_type == "segmentation_mask"
                    else predictions
                )
                reference_image_data = (
                    self._masks_to_pixels(test_output)
                    if self.output_type == "segmentation_mask"
                    else test_output
                )
                input_images = [
                    wandb.Image(data, grouping=3)
                    for i, data in enumerate(input_image_data)
                ]
                output_images = [
                    wandb.Image(data) for i, data in enumerate(output_image_data)
                ]
                reference_images = [
                    wandb.Image(data) for i, data in enumerate(reference_image_data)
                ]
                return list(
                    chain.from_iterable(
                        zip(input_images, output_images, reference_images)
                    )
                )
            else:
                # unknown output, just log the input images
                return [wandb.Image(img) for img in test_data]
        elif self.output_type in ("image", "images", "segmentation_mask"):
            # unknown input, just log the predicted and reference outputs without captions
            output_image_data = (
                self._masks_to_pixels(predictions)
                if self.output_type == "segmentation_mask"
                else predictions
            )
            reference_image_data = (
                self._masks_to_pixels(test_output)
                if self.output_type == "segmentation_mask"
                else test_output
            )
            output_images = [
                wandb.Image(data, grouping=2)
                for i, data in enumerate(output_image_data)
            ]
            reference_images = [
                wandb.Image(data) for i, data in enumerate(reference_image_data)
            ]
            return list(chain.from_iterable(zip(output_images, reference_images)))

    def _log_weights(self):
        metrics = {}
        for layer in self.model.layers:
            weights = layer.get_weights()
            if len(weights) == 1:
                _update_if_numeric(
                    metrics, "parameters/" + layer.name + ".weights", weights[0]
                )
            elif len(weights) == 2:
                _update_if_numeric(
                    metrics, "parameters/" + layer.name + ".weights", weights[0]
                )
                _update_if_numeric(
                    metrics, "parameters/" + layer.name + ".bias", weights[1]
                )
        return metrics

    def _log_gradients(self):
        # Suppress callback warnings grad accumulator
        og_level = tf_logger.level
        tf_logger.setLevel("ERROR")

        self._grad_accumulator_model.fit(
            self._training_data_x,
            self._training_data_y,
            verbose=0,
            callbacks=[self._grad_accumulator_callback],
        )
        tf_logger.setLevel(og_level)
        weights = self.model.trainable_weights
        grads = self._grad_accumulator_callback.grads
        metrics = {}
        for (weight, grad) in zip(weights, grads):
            metrics[
                "gradients/" + weight.name.split(":")[0] + ".gradient"
            ] = wandb.Histogram(grad)
        return metrics

    def _log_dataframe(self):
        x, y_true, y_pred = None, None, None

        if self.validation_data:
            x, y_true = self.validation_data[0], self.validation_data[1]
            y_pred = self.model.predict(x)
        elif self.generator:
            if not self.validation_steps:
                wandb.termwarn(
                    "when using a generator for validation data with dataframes, you must pass validation_steps. skipping"
                )
                return None

            for i in range(self.validation_steps):
                bx, by_true = next(self.generator)
                by_pred = self.model.predict(bx)
                if x is None:
                    x, y_true, y_pred = bx, by_true, by_pred
                else:
                    x, y_true, y_pred = (
                        np.append(x, bx, axis=0),
                        np.append(y_true, by_true, axis=0),
                        np.append(y_pred, by_pred, axis=0),
                    )

        if self.input_type in ("image", "images") and self.output_type == "label":
            return wandb.image_categorizer_dataframe(
                x=x, y_true=y_true, y_pred=y_pred, labels=self.labels
            )
        elif (
            self.input_type in ("image", "images")
            and self.output_type == "segmentation_mask"
        ):
            return wandb.image_segmentation_dataframe(
                x=x,
                y_true=y_true,
                y_pred=y_pred,
                labels=self.labels,
                class_colors=self.class_colors,
            )
        else:
            wandb.termwarn(
                "unknown dataframe type for input_type=%s and output_type=%s"
                % (self.input_type, self.output_type)
            )
            return None

    def _save_model(self, epoch):
        if wandb.run.disabled:
            return
        if self.verbose > 0:
            print(
                "Epoch %05d: %s improved from %0.5f to %0.5f,"
                " saving model to %s"
                % (epoch, self.monitor, self.best, self.current, self.filepath)
            )

        try:
            if self.save_weights_only:
                self.model.save_weights(self.filepath, overwrite=True)
            else:
                self.model.save(self.filepath, overwrite=True)
        # Was getting `RuntimeError: Unable to create link` in TF 1.13.1
        # also saw `TypeError: can't pickle _thread.RLock objects`
        except (ImportError, RuntimeError, TypeError) as e:
            wandb.termerror(
                "Can't save model in the h5py format. The model will be saved as "
                "W&B Artifacts in the SavedModel format."
            )
            self.save_model = False
            self.save_model_as_artifact = True

    def _save_model_as_artifact(self, epoch):
        if wandb.run.disabled:
            return

        # Save the model in the SavedModel format.
        # TODO: Replace this manual artifact creation with the `log_model` method
        # after `log_model` is released from beta.
        self.model.save(self.filepath[:-3], overwrite=True, save_format="tf")

        # Log the model as artifact.
        model_artifact = wandb.Artifact(f"model-{wandb.run.name}", type="model")
        model_artifact.add_dir(self.filepath[:-3])
        wandb.run.log_artifact(model_artifact, aliases=["latest", f"epoch_{epoch}"])

        # Remove the SavedModel from wandb dir as we don't want to log it to save memory.
        shutil.rmtree(self.filepath[:-3])