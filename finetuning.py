from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader, random_split

import torchvision.transforms.v2 as T
import torchvision.datasets as datasets
import torchvision.models as tv_models

import pandas as pd
import numpy as np
from PIL import Image

from tqdm.auto import tqdm
from IPython.display import display, Markdown

import matplotlib.pyplot as plt

import math
import random
from pathlib import Path
from typing import Any, Sequence


class FinetuningWeights:
    """
    These are the weights for the model that we are going to fine-tune.
    Like other classes here, the base model is assumed to be MobileNetV3.
    So, some things are very specific to this model and the available weights.

    It is not a very general set of classes (like Lightning) that work for a
    much broader set of models.

    Args:
        model_name: DEFAULT eights are obtained for this model name.
    """

    def __init__(self, model_name: str = "mobilenet_v3_large") -> None:
        self.model_name = model_name
        self.weights = tv_models.get_model_weights("mobilenet_v3_large").DEFAULT
        self.transforms = self.weights.transforms()

        self.metrics: dict = {}
        self.log()
        self.print_metrics()

    @property
    def mean(self) -> list[float]:
        """
        Mean for the training dataset used (ImageNet).

        Returns:
            3 floats corresponding to RGB.
        """
        return self.transforms.mean

    @property
    def std(self) -> list[float]:
        """
        Standard Deviation for the training dataset used (ImageNet).

        Returns:
            3 floats corresponding to RGB.
        """
        return self.transforms.std

    @property
    def image_size(self) -> list[int]:
        """
        Final size of the images used in training.

        Returns:
            List of 1 int, which is both height and width
        """
        return self.transforms.crop_size

    # -------
    # Logging
    # -------

    def log(self) -> None:
        """Metrics on the weights."""

        self.metrics["name"] = self.weights.name
        self.metrics["url"] = self.weights.url
        self.metrics["num categories"] = len(self.weights.meta["categories"])

        self.metrics["mean"] = self.mean
        self.metrics["std"] = self.std
        self.metrics["image size"] = self.image_size

    def print_metrics(self) -> None:
        display(Markdown("*`FinetuningWeights` is initialized. Metrics:*"))
        df = pd.DataFrame(self.metrics.items(), columns=["metric", "value"])
        display(df.style.hide(axis="index"))


class FinetuningDataModule:
    """
    Provides the data that is needed for this fine-tuning exercise. Datasets are
    downloaded from EMNIST and the train, val, and test datasets and dataloaders
    are made. For this class, the `digits` split is chosen. So, the images form
    10 classes for 10 digits (0 through 9).

    Args:
        root: the folder/path where the downloaded dataset files are stored.
        weights: a FinetuningWeights object that has mean, std, etc. values for
            the datasets.
        train_val_split: Split between training and validation datasets
        batch_size: for the dataloaders
    """

    def __init__(
        self,
        root: str | Path,
        weights: FinetuningWeights,
        train_val_split: Sequence[int | float] = (0.8, 0.2),
        batch_size: int = 256,
    ) -> None:
        # Passed parameters
        self.root = root
        self.train_val_split = train_val_split
        self.weights = weights
        self.batch_size = batch_size

        # Datasets
        self.train_dataset: Dataset = None
        self.val_dataset: Dataset = None
        self.test_dataset: Dataset = None

        # Data Loaders
        self.train_dataloader: DataLoader = None
        self.val_dataloader: DataLoader = None
        self.test_dataloader: DataLoader = None

        self.setup()

        # Metrics
        self.metrics: dict[str, Any] = {}
        self.log()
        self.print_metrics()

    def setup(self):
        """Prepare datasets and dataloaders"""
        self.prepare_datasets()
        self.prepare_dataloaders()

    def prepare_datasets(self):
        """
        Datasets are downloaded from torchvision (in this case EMNIST).
        Main training dataset is split into training and validation.
        Separate test dataset (non-training) is also downloaded.
        """
        # Training and validation datasets
        train_val_dataset = datasets.EMNIST(
            self.root, split="digits", train=True, download=True
        )

        # Split main dataset into train and validation datasets
        train_dataset_untransformed, val_dataset_untransformed = random_split(
            dataset=train_val_dataset, lengths=self.train_val_split
        )

        # Apply transformations separately to train and val
        self.train_dataset = TransformedDataset(
            dataset=train_dataset_untransformed,
            transform=self.train_transforms(),
        )
        self.val_dataset = TransformedDataset(
            dataset=val_dataset_untransformed, transform=self.eval_transforms()
        )

        # Download test dataset and apply transformations
        # Test dataset is not further split, so, we can apply transformations
        # right at the download.
        self.test_dataset = datasets.EMNIST(
            self.root,
            split="digits",
            train=False,
            download=True,
            transform=self.eval_transforms(),
        )

    def prepare_dataloaders(self):
        """
        Prepares train, val, and test dataloaders.
        Train dataloader shuffles the samples. Val and Test dataloaders don't.
        batch_size passed to the class is used.
        """
        self.train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )
        self.val_dataloader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )
        self.test_dataloader = DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )

    # ---------------
    # Transformations
    # ---------------

    def basic_transforms_list(self) -> list[Any]:
        """
        List of transformations to be applied all the incoming images
        (these are necessary transformations, does not include augmentations):

        - Grayscale: 1 channel gray digits will become 3-channel (RGB)
            gray digits.
        - Resize: Image resized to the size used in original model training
        - ToImage: Convert PIL image to Tensor
        - Lambda: transposes the image - a correction on the origninal
            MNIST images.
        - ToDtype: ToImage gives values in the range of 0-255 for pixels.
            Here, they are scaled to a number between 0.0 and 1.0 (a float32).
        - Normalize: Normalizes the tensor images with dataset mean and
            standard deviation.

        Returs:
            List of transformations
        """
        # Get mean, std, and image size from the weights
        return [
            T.Grayscale(num_output_channels=3),
            T.Resize(size=self.weights.image_size),
            T.ToImage(),
            T.Lambda(lambda img: img.transpose(-2, -1)),
            T.ToDtype(dtype=torch.float32, scale=True),
            T.Normalize(self.weights.mean, self.weights.std),
        ]

    def augmentation_transforms_list(self) -> list[Any]:
        """
        Some random transformations are applied to the training set of images.
        This helps model learn more because it sees subtle changes in the same
        original image.

        For MNIST dataset, augmentations are not used (explanation provided
            elsewhere).
        Just listed here for completeness.
        """
        return [
            T.RandomRotation(degrees=5),
            T.RandomAdjustSharpness(sharpness_factor=1.5, p=0.3),
            T.ColorJitter(brightness=0.1, contrast=0.1),
            T.RandomVerticalFlip(p=0.5),
            T.RandomAffine(
                degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)
            ),
        ]

    def train_transforms(self) -> T.Compose:
        """
        Provides composed transforms to be used on training dataset.

        Typically, the entire list is made like this (augmentation+basic):
        `T.Compose(self.augmentation_transforms_list+self.basic_transforms_list)`

        However, in case of MNIST dataset, no augmentations are used.
        (explanation provided elsewhere)
        """
        return T.Compose(self.basic_transforms_list())

    def eval_transforms(self) -> T.Compose:
        """
        Provides a composed transforms to be used in evaluation datasets:

        - validation dataset (in the training loop)
        - test dataset (for post-training)

        Returns:
            Composed transforms for eval.
        """
        return T.Compose(self.basic_transforms_list())

    # -------
    # Logging
    # -------

    def log(self) -> None:
        """Metrics on the data."""

        self.metrics["root"] = self.root
        self.metrics["batch_size"] = self.batch_size
        self.metrics["train_val_split"] = self.train_val_split

        self.metrics["train dataset samples"] = f"{len(self.train_dataset):,}"
        self.metrics["val dataset samples"] = f"{len(self.val_dataset):,}"
        self.metrics["test dataset samples"] = f"{len(self.test_dataset):,}"

    def print_metrics(self) -> None:
        """Prints metrics."""
        display(Markdown("*`FinetuningDataModule` is initialized. Metrics:*"))
        df = pd.DataFrame(self.metrics.items(), columns=["metric", "value"])
        display(df.style.hide(axis="index"))


class TransformedDataset(Dataset):
    """
    Specified transformation is applied as a sample is retrieved at a given
    index.

    Typically, transformations are applied when the original dataset is made.
    However, in this case, the same dataset is split into train and validation
    datasets by using random_split. This method yeilds two separate Subsets that
    share the same dataset in the back. Since transformations are different
    between, a separate class should apply individual transformations to each
    dataset. Hence the need for this TransformedDataset.

    Additional Note: In this specific case (MNIST) though no augmentation
    transformations are applied on train dataset. So, this class doesn't
    doesn't add anything (for this specific case). It is shown for completeness
    (and for the cases where transformations are different between training and
    validation datasets)

    Args:
        dataset: train or validation dataset
        transform: the transformation to be applied for that specific dataset
    """

    def __init__(self, dataset: Dataset, transform: T.Transform) -> None:
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """
        Returns one sample at the provided index.

        Args:
            index: of the current sample.

        Returns:
            A tuple of (transformed image tensor and integer label for the
            class of the image)
        """
        image, label = self.dataset[index]
        image_transformed = self.transform(image)
        return image_transformed, label

    def __len__(self) -> int:
        """Number of samples in the dataset."""
        return len(self.dataset)


class FinetuningModule:
    """
    Base model that is being fine-tuned is downloaded and setup for fine-tuning.

    Args:
        weights: contains the name of the base model.
        new_out_featues: how many classes are there in the new fine-tune data.
            The original MobileNetV3 has 1000 classes; if this is being
            fine-tuned for digits data, the number of out features will be 10.
            This is used in the last Linear layer of the model classifier.
        num_layers_to_tune: supports different fine-tuning scenarios:
            - 0: classifier: only the classifier is changed and re-trained.
                Remaining weights are not changed.
            - 1-16: classifier+: In addition to new classifier, a specified
                number of layers are retrained.
            - -1: entire model: All layers in the model are re-trained.
                Training starts with the weights that are there in the original
                model; but all of them are changed during training.
        learning_rate: for the optimizer.
    """

    def __init__(
        self,
        weights: FinetuningWeights,
        new_out_features: int = 10,
        num_layers_to_tune: int = 0,
        learning_rate: float = 1e-3,
    ) -> None:
        self.model_name = weights.model_name
        self.model: nn.Module = None

        self.new_out_features = new_out_features
        self.num_layers_to_tune = num_layers_to_tune
        self.learning_rate = learning_rate

        self.loss_fn: Any = None
        self.optimizer: Any = None

        self.metrics: dict = {}

        self.setup()
        self.log()
        self.print_metrics()

    # ----------
    # Base Model
    # ----------

    def setup(self):
        """Download the model and setup the architecture for fine-tuning."""
        self.model = self.download_model()

        self.original_model_parameters = (
            f"{sum(p.numel() for p in self.model.parameters()):,}"
        )
        self.original_model_feature_parameters = (
            f"{sum(p.numel() for p in self.model.features.parameters()):,}"
        )
        self.original_model_classifier_parameters = (
            f"{sum(p.numel() for p in self.model.classifier.parameters()):,}"
        )

        self.forward()

    def download_model(self):
        """Download the specified base model."""
        return tv_models.get_model(
            name=self.model_name, weights="DEFAULT", progress=True
        )

    def forward(self):
        """Fine-tune the architecture according the input scenario."""
        self.fine_tune_architecture()

    # ------------------------------------
    # Architecture Changes for fine-tuning
    # ------------------------------------

    def fine_tune_architecture(self) -> None:
        """
        Fine-tune the model architecture:
            - turn-off all features parameters
            - add new classifier
            - turn on some or none of features layers parameters
            - create loss and optim for the params to be trained
        """
        self.turn_off_gradients_all_features_layers()
        self.add_new_final_classifier_layer()
        self.turn_on_gradients_for_selected_layers()
        self.create_loss_optim()

    def turn_off_gradients_all_features_layers(self) -> None:
        for param in self.model.features.parameters():
            param.requires_grad = False

    def add_new_final_classifier_layer(self) -> None:
        """
        A new classifier layer is added with the given number of output classes.
        """
        # Final Linear layer of the classifier
        final_layer = self.model.classifier[-1]
        # Keep the input features coming into this final layer
        in_features = final_layer.in_features

        # Create new final classifier layer
        new_classifier_output_layer = nn.Linear(
            in_features=in_features,
            out_features=self.new_out_features,
            bias=True,
        )
        # Replace final classifier layer
        self.model.classifier[-1] = new_classifier_output_layer

    def turn_on_gradients_for_selected_layers(self):
        """
        All model layers are tuned, if num_layers_to_tune is -1.
        Otherwise, only the last num_layers_to_tune feature layers are tuned.
        """
        if self.num_layers_to_tune == -1:
            for param in self.model.parameters():
                param.requires_grad = True
        else:
            for i in range(self.num_layers_to_tune):
                for param in self.model.features[-i].parameters():
                    param.requires_grad = True

    def create_loss_optim(self):
        """
        Loss function is CrossEntropyLoss for this classification task.
        Optimizer is SGD for the parameters that require gradient adjustments (fine-tuning)

        Returns:
            Loss function and optimizer.
        """
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.learning_rate,
        )

    # -------
    # Logging
    # -------

    def log(self) -> None:
        """Logs various metrics for this class."""
        """Logs various metrics."""
        finetune_model_total_parameters = (
            f"{sum(p.numel() for _, p in self.model.named_parameters()):,}"
        )
        num_feature_parameters = (
            f"{sum(p.numel() for p in self.model.features.parameters()):,}"
        )
        num_final_classifier_parameters = (
            f"{sum(p.numel() for p in self.model.classifier.parameters()):,}"
        )

        num_parameters_to_tune = f"{
            sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            ):,}"
        num_parameters_unchanged = f"{
            sum(
                p.numel()
                for p in self.model.parameters()
                if not p.requires_grad
            ):,}"

        self.metrics["model name"] = self.model_name
        self.metrics["num feature extractor blocks"] = len(self.model.features)
        self.metrics["num classifier layers"] = len(self.model.classifier)

        self.metrics["original model - total parameters"] = (
            self.original_model_parameters
        )
        self.metrics["original model - num feature parameters"] = (
            self.original_model_feature_parameters
        )
        self.metrics["original model - num classifier parameters"] = (
            self.original_model_classifier_parameters
        )

        self.metrics["fine-tune model - total parameters"] = (
            finetune_model_total_parameters
        )
        self.metrics["fine-tune model - num feature parameters"] = (
            num_feature_parameters
        )
        self.metrics["fine-tune model - num classifier parameters"] = (
            num_final_classifier_parameters
        )

        self.metrics["fine-tune model - num parameters to tune"] = (
            num_parameters_to_tune
        )
        self.metrics["fine-tune model - num parameters unchanged"] = (
            num_parameters_unchanged
        )

    def print_metrics(self) -> None:
        """Prints metrics."""
        display(Markdown("*`FinetuningModule` is initialized. Metrics:*"))
        df = pd.DataFrame(self.metrics.items(), columns=["metric", "value"])
        display(df.style.hide(axis="index"))

    # -----------------------------
    # Training, Validation, Testing
    # -----------------------------

    def training_step(self, inputs: Any, labels: Any) -> Any:
        """
        One batch of samples are used to predict, calculate loss, backpropagate, and adjust
        parameters.

        Retuns:
            Loss: for this batch.
        """
        self.model.train()

        self.optimizer.zero_grad()
        preds = self.model(inputs)

        loss = self.loss_fn(preds, labels)
        loss.backward()
        self.optimizer.step()

        return loss

    def validation_step(self, inputs: Any, labels: Any) -> Any:
        """
        Validation phase predictions are made for one batch of samples. Correct
        number of predictions is calculated.

        Returns:
            Number of correct predictions
        """
        return self.evaluation_step(inputs=inputs, labels=labels)

    def testing_step(self, inputs: Any, labels: Any) -> Any:
        """
        Testing phase predictions are made for one batch of samples. Correct
        number of predictions is calculated.

        Returns:
            Number of correct predictions
        """
        return self.evaluation_step(inputs=inputs, labels=labels)

    def evaluation_step(self, inputs: Any, labels: Any) -> Any:
        """
        Common method for Validation and Testing phase predictions for one
        batch of samples. Correct number of predictions is calculated.

        Returns:
            Number of correct predictions
        """
        self.model.eval()

        correct: int = 0
        with torch.no_grad():
            preds = self.model(inputs)
            _, pred_labels = preds.max(1)
            correct = (pred_labels == labels).sum()

        return correct


class FinetuningTrainer:
    """
    Fine-tunes a base model with the provided new data. Includes fit - an
    orchestration function that calls training, validation, and testing steps
    to finish the training. Captures metrics.

    Args:
        module: A FinetuningModule that provides model and fine-tuning steps
        data: A FinetuningDataModule that provides data loaders
        epochs: Number of loops for the fine-tuning training and validation.
    """

    def __init__(
        self,
        experiment: str,
        analytics: FinetuningAnalytics,
        module: FinetuningModule,
        data: FinetuningDataModule,
        epochs: int = 1,
    ) -> None:
        self.epochs = epochs
        self.module = module
        self.model = module.model

        self.data = data
        self.train_dataloader = self.data.train_dataloader
        self.val_dataloader = self.data.val_dataloader
        self.test_dataloader = self.data.test_dataloader

        self.metrics: dict = {}
        self.device = self.choose_device()

        self.experiment = experiment
        self.analytics = analytics
        self.analytics.experiments[experiment] = self

    def fit(self) -> nn.Module:
        """
        Actual training/fine-tuning loop. Training and validation are performed
        for the given number of epochs. Loss and accuracy are calculated during
        these epochs. After a model is trained, a testing loop is done for the
        accuracy with test data.

        Returns:
            fine-tuned model. This is also available via the instance of this
                class.
        """
        self.model.to(self.device)

        # Epochs
        loss_final = 0.0
        accuracy_final = 0.0

        for epoch in range(self.epochs):
            # Training
            loss_epoch = 0.0

            pbar_train = tqdm(
                iterable=self.train_dataloader,
                desc="training",
                unit="batches",
                leave=False,
            )
            loss_epoch_avg = 0.0
            for idx_train, (inputs, labels) in enumerate(pbar_train):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                loss_step = self.module.training_step(
                    inputs=inputs, labels=labels
                )
                batch_loss = loss_step.item()
                loss_epoch += batch_loss
                loss_epoch_avg = loss_epoch / (idx_train + 1)
                pbar_train.set_postfix(
                    {
                        "batch loss": f"{batch_loss:.4f}",
                        "avg loss": f"{loss_epoch_avg:.4f}",
                    }
                )

            self.log(
                metric=f"epoch{epoch}: training loss", value=loss_epoch_avg
            )
            loss_final += loss_epoch

            # Validation
            correct_epoch = 0

            pbar_val = tqdm(
                iterable=self.val_dataloader,
                desc="validation",
                unit="batches",
                leave=False,
            )
            for idx_val, (inputs, labels) in enumerate(pbar_val):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                correct_step = self.module.validation_step(
                    inputs=inputs, labels=labels
                )
                correct_epoch += correct_step.item()
                out_of = (idx_val + 1) * self.val_dataloader.batch_size
                pbar_val.set_postfix(
                    {
                        "correct": correct_epoch,
                        "out of": out_of,
                    }
                )

            accuracy_epoch = correct_epoch / len(self.val_dataloader.dataset)
            accuracy_final += accuracy_epoch
            self.log(
                metric=f"epoch {epoch}: validation accuracy",
                value=accuracy_epoch,
            )

            print(
                f"Epoch {epoch}: loss = {loss_epoch_avg:.4f}, accuracy = {accuracy_epoch * 100:.2f}%"
            )

        print(f"\nTraining and Validation Complete for: {self.experiment}.\n")

        # Testing
        testing_correct_total = 0

        pbar_test = tqdm(
            iterable=self.test_dataloader,
            desc="testing",
            unit="batches",
            leave=False,
        )
        # total=num_test_batches
        for inputs, labels in pbar_test:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            testing_correct_step = self.module.testing_step(
                inputs=inputs, labels=labels
            )
            testing_correct_step = testing_correct_step.item()
            testing_correct_total += testing_correct_step
            pbar_test.set_postfix(
                {
                    "correct": testing_correct_step,
                    "batch size": self.test_dataloader.batch_size,
                }
            )

        # Metrics at the end of testing
        accuracy_testing = testing_correct_total / len(
            self.test_dataloader.dataset
        )
        self.log(metric="Final: testing accuracy", value=accuracy_testing)
        print(f"Final: testing accuracy = {accuracy_testing * 100:.2f}%")

        print(f"\nTesting Complete for: {self.experiment}.\n")

        self.print_metrics()

        pbar_train.clear()
        pbar_test.clear()
        pbar_val.clear()
        pbar_train.close()
        pbar_test.close()
        pbar_val.close()
        pbar_train = None
        pbar_test = None
        pbar_val = None

        return self.model

    def choose_device(self, seed: int = 42) -> torch.device:
        """Returns the best device available."""
        torch.manual_seed = seed
        device = torch.device("cpu")

        if torch.cuda.is_available():
            device = torch.device("cuda")
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            torch.mps.manual_seed = seed

        self.log(metric="device", value=device)
        return device

    # -------
    # Logging
    # -------

    def log(self, metric: str, value: Any) -> None:
        """Logs and prints the metrics as training/validation/testing occurs."""
        self.metrics[metric] = value
        # print(f"{metric} = {value}")

    def print_metrics(self) -> None:
        """Prints metrics."""
        display(Markdown("*`FinetuningTrainier` fit() is completed. Metrics:*"))
        df = pd.DataFrame(self.metrics.items(), columns=["metric", "value"])
        display(df.style.hide(axis="index"))


class FinetuningAnalytics:
    """
    This class has miscellaneous plots and prediction display:

    1. Image Grid with predictions for random samples
    2. Image Grid with wrong predictions for random samples
    3. Loss-Accuracy plot for a given experiment
    4. Accuracy comparison across experiments
    """

    def __init__(self) -> None:
        # Experiments are trainers and they add themselves to the following list
        self.experiments: dict[str, FinetuningTrainer] = {}

    # -----------
    # Image Grids
    # -----------

    def test_predictions(self, experiment: str) -> None:
        """
        10 random samples are chosen from the test dataset of the experiment.
        The results (label, image, prediction) are shown in an image grid.
        """
        ds = self.experiments[experiment].test_dataloader.dataset
        display(Markdown(f"*Random Predictions from **{experiment}**:*"))

        max_idx = len(ds) - 1
        random.seed(42)

        idxs = [random.randint(0, max_idx) for i in range(10)]
        samples = [ds[idx] for idx in idxs]
        imgs = [img for img, _ in samples]
        labels = [label for _, label in samples]

        preds_t = self.predict(experiment, imgs)
        preds = preds_t.tolist()
        sample_preds = []
        for label, img, pred in zip(labels, imgs, preds):
            sample_pred = {"label": label, "img": img, "pred": pred}
            sample_preds.append(sample_pred)

        self.show_img_grid(sample_preds=sample_preds)

    def test_wrong_predictions(self, experiment: str):
        """
        Random samples are sent to prediction until 10 wrong predictions are
        accumulated. These results (label, image, prediction) are shown in an
        image grid.

        This wrong prediction grid shows what kind of images that the model is
        having trouble with.
        """
        ds = self.experiments[experiment].test_dataloader.dataset
        display(Markdown(f"*Random Wrong Predictions from **{experiment}**:*"))

        random.seed(42)

        wrong_preds: list[dict[str, Any]] = []
        self.get_wrong_preds(experiment, ds, wrong_preds)

        self.show_img_grid(wrong_preds)

    def test_wrong_predictions_across_experiments(self, idx_start: int = 1000):
        """
        Random samples are sent to prediction until 10 wrong predictions are
        accumulated. These results (label, image, prediction) are shown in an
        image grid.

        This wrong prediction grid shows what kind of images that the model is
        having trouble with.
        """
        display(
            Markdown(
                f"*Random Wrong Predictions from **{list(self.experiments.keys())}**:*"
            )
        )

        random.seed(42)

        wrong_preds = self.get_wrong_preds_across_experiments(idx_start)
        self.show_img_grid(wrong_preds)

    def get_wrong_preds_from_idx(
        self,
        experiment: str,
        ds: Dataset,
        wrong_preds: list[dict[str, Any]],
        idx_start: int,
        batch_size: int = 256,
    ) -> None:
        """
        Go through samples from the start index for a batch and provide details of
        wrong predictions.
        """
        idxs = [i for i in range(idx_start, idx_start + batch_size)]
        samples = [ds[idx] for idx in idxs]
        labels = [label for _, label in samples]
        imgs = [img for img, _ in samples]

        preds = self.predict(experiment, imgs)

        for label, img, pred, idx in zip(labels, imgs, preds, idxs):
            if label != pred:
                sample_pred = {
                    "label": label,
                    "img": img,
                    "pred": pred.item(),
                    "idx": idx,
                }
                wrong_preds.append(sample_pred)

    def get_wrong_preds_across_experiments(
        self,
        idx_start: int,
    ) -> list[dict[str, Any]]:
        """
        Gives a list of 10 wrong predictions by all the models in the experiment;
        and this search for wrong predictions start at the index specified.

        Args:
            idx_start: The start index in test dataset to search for wrong preds

        Returns:
            List of wrong predictions
        """
        first_exp = list(self.experiments.keys())[0]
        ds = self.experiments[first_exp].test_dataloader.dataset
        common_wrong_preds: list[dict[str, Any]] = []

        while len(common_wrong_preds) < 10:
            wrong_preds_list = []
            for experiment in self.experiments.keys():
                experiment_wrong_preds: list[dict[str, Any]] = []
                self.get_wrong_preds_from_idx(
                    experiment, ds, experiment_wrong_preds, idx_start
                )
                wrong_preds_list.append(experiment_wrong_preds)

            wrong_idxs_list = []
            for wrong_preds in wrong_preds_list:
                wrong_idxs = [p["idx"] for p in wrong_preds]
                wrong_idxs_list.append(wrong_idxs)

            common_wrong_idxs_set = set(wrong_idxs_list[0])
            for i in range(1, len(wrong_idxs_list)):
                common_wrong_idxs_set = common_wrong_idxs_set & set(
                    wrong_idxs_list[i]
                )

            common_wrong_idxs = sorted(common_wrong_idxs_set)
            for idx in common_wrong_idxs:
                samples = []
                common_wrong_pred = {}
                for wrong_preds in wrong_preds_list:
                    sample = [s for s in wrong_preds if s.get("idx") == idx]
                    samples.append(sample[0])

                common_wrong_pred = {
                    "label": samples[0]["label"],
                    "img": samples[0]["img"],
                    "pred": [p["pred"] for p in samples],
                }

                if common_wrong_pred and len(common_wrong_preds) < 10:
                    common_wrong_preds.append(common_wrong_pred)

            idx_start += 256

        return common_wrong_preds

    # --------------------------
    # Image Grid Support Methods
    # --------------------------

    def predict(self, experiment: str, imgs: Any):
        """Predictions for given images."""
        model = self.experiments[experiment].model
        device = self.experiments[experiment].device
        imgs_t = torch.stack(imgs, dim=0)

        model = model.to(device)
        model.eval()
        with torch.no_grad():
            imgs_t = imgs_t.to(device)
            class_preds = model(imgs_t)
            preds = class_preds.argmax(dim=1)

        return preds

    def show_img_grid(self, sample_preds: list[dict[str, Any]]):
        """Display image grid with label, img, prediction."""
        fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(8, 4))
        for idx, ax in enumerate(axes.flatten()):
            label, img, pred = (
                sample_preds[idx - 1]["label"],
                sample_preds[idx - 1]["img"],
                sample_preds[idx - 1]["pred"],
            )

            ax.axis("off")
            img = img.detach().cpu().permute(1, 2, 0).clamp(0, 1)
            ax.imshow(img, cmap="gray")

            if label == pred:
                pred_color = "green"
            else:
                pred_color = "red"
            ax.text(
                0,
                1.05,
                f"label={label}",
                color="blue",
                transform=ax.transAxes,
            )
            ax.text(
                0,
                -0.15,
                f"pred={pred}",
                color=pred_color,
                transform=ax.transAxes,
            )

        plt.show()

    def get_wrong_preds(
        self, experiment: str, ds: Dataset, wrong_preds: list[dict[str, Any]]
    ) -> None:
        """
        Go through samples, until 10 wrong predictions are obtained.
        """
        max_idx = len(ds) - 1

        while len(wrong_preds) < 10:
            # Get 100 indices and make predictions at a time
            idxs = [random.randint(0, max_idx) for i in range(100)]
            samples = [ds[idx] for idx in idxs]
            labels = [label for _, label in samples]
            imgs = [img for img, _ in samples]

            preds = self.predict(experiment, imgs)

            for label, img, pred in zip(labels, imgs, preds):
                if label != pred:
                    sample_pred = {"label": label, "img": img, "pred": pred}
                    wrong_preds.append(sample_pred)
                if len(wrong_preds) == 10:
                    break

    # -----
    # Plots
    # -----

    def loss_accuracy_plot(self, experiment: str) -> None:
        """
        Plot between training loss and validation accuracy for a given
        experiment.

        Args:
            experiment: Name of the experiment, e.g. fine-tune classifier.
        """
        metrics = self.experiments[experiment].metrics
        display(
            Markdown(
                f"*Training Loss and Validation Accuracy from **{experiment}**:*"
            )
        )

        loss, accuracy, epochs = self.get_loss_accuracy(metrics)

        fig, ax = plt.subplots(figsize=(6, 4))

        ax.plot(epochs, loss, color="red", marker="s", label="Training Loss")
        ax.plot(
            epochs,
            accuracy,
            color="green",
            marker="o",
            label="Validation Accuracy",
        )

        ax.set_title("Training Loss vs. Validation Accuracy")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss/Accuracy")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.5)

        plt.tight_layout()
        plt.show()

    def accuracy_compare_plot(self) -> None:
        """Plot that compares the accuracies between experiments."""
        acc_list = []
        epoch_list = []
        labels = []
        for trainer in self.experiments.values():
            _, accuracy, epoch = self.get_loss_accuracy(trainer.metrics)
            acc_list.append(accuracy * 100)
            epoch_list.append(epoch)
            labels.append(trainer.experiment)

        display(
            Markdown(
                f"*Accuracy Comparison between experiments: **{list(self.experiments.keys())}**:*"
            )
        )
        fig, ax = plt.subplots(figsize=(8, 6))
        for i in range(len(labels)):
            ax.plot(
                epoch_list[i],
                acc_list[i],
                color=f"C{i}",
                marker="o",
                label=labels[i],
            )
        ax.set_title("Accuracy Comparison across Experiments")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.5)

        plt.tight_layout()
        plt.show()

    # ------------
    # Plot Support
    # ------------

    def get_loss_accuracy(
        self, metrics: dict[str, float]
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """
        Splits the metrics dictionary into training loss and validation accuracy.

        Args:
            metrics: dictionary from the training run.

        Returns:
            loss, accuracy, epochs: training loss, validation accuracy, and
                the epochs list.
        """
        df = pd.DataFrame(metrics.items(), columns=["metric", "value"])

        loss = df[df["metric"].str.contains("training loss")][
            "value"
        ].to_numpy()
        accuracy = df[df["metric"].str.contains("validation accuracy")][
            "value"
        ].to_numpy()
        epochs = np.arange(1, len(loss) + 1)

        return loss, accuracy, epochs


if __name__ == "__main__":
    experiment = "fine-tune classifier"
    analytics = FinetuningAnalytics()

    weights = FinetuningWeights()
    data = FinetuningDataModule(root="./data/finetune", weights=weights)
    backbone = FinetuningModule(weights=weights)

    trainer = FinetuningTrainer(
        module=backbone, data=data, experiment=experiment, analytics=analytics
    )
    finetuned_model = trainer.fit()
    analytics.test_predictions(experiment=experiment)
