import torch
import requests
import numpy as np
import supervision as sv
import albumentations as A

from PIL import Image
from roboflow import Roboflow

from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
    TrainingArguments,
    Trainer
)
from dataclasses import replace
from methods.pytorch_detection_dataset import PyTorchDetectionDataset
from methods.mape_evaluator import MAPEvaluator

class ModelTrainer:
    def __init__(self, checkpoint, roboflow_api_key, device=None):
        self.checkpoint = checkpoint
        self.roboflow_api_key = roboflow_api_key
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForObjectDetection.from_pretrained(self.checkpoint).to(self.device)
        IMAGE_SIZE = 480
        self.processor = AutoImageProcessor.from_pretrained(self.checkpoint, do_resize=True, size={"width": IMAGE_SIZE, "height": IMAGE_SIZE})

        # Augmentation with bbox validation
        self.train_augmentation_and_transform = A.Compose(
            [
                A.Perspective(p=0.1),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.HueSaturationValue(p=0.1),
            ],
            bbox_params=A.BboxParams(
                format="coco",
                label_fields=["category"],
                clip=True,
                check_each_transform=True,  # Ensure bbox is checked after each transform
                min_area=25  # Ignore very small bounding boxes
            ),
        )

        self.valid_transform = A.Compose(
            [A.NoOp()],
            bbox_params=A.BboxParams(
                format="coco",
                label_fields=["category"],
                clip=True,
                min_area=1
            ),
        )

    def filter_invalid_bboxes(self, bboxes):
        """
        Filter out invalid bounding boxes where x_min >= x_max or y_min >= y_max
        """
        valid_bboxes = []
        for bbox in bboxes:
            x_min, y_min, x_max, y_max = bbox
            if x_min < x_max and y_min < y_max:
                valid_bboxes.append(bbox)
            else:
                print(f"Invalid bbox removed: {bbox}")
        return valid_bboxes

    def run_inference(self, url):
        image = Image.open(requests.get(url, stream=True).raw)
        inputs = self.processor(image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        w, h = image.size
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=[(h, w)], threshold=0.3)

        detections = sv.Detections.from_transformers(results[0]).with_nms(threshold=0.1)
        labels = [self.model.config.id2label[class_id] for class_id in detections.class_id]

        annotated_image = image.copy()
        annotated_image = sv.BoundingBoxAnnotator().annotate(annotated_image, detections)
        annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels=labels)
        annotated_image.thumbnail((600, 600))

        return annotated_image
    
    def evaluate_model(self, ds_test):
        print("Evaluating model...")

        targets = []
        predictions = []

        for i in range(len(ds_test)):
            path, source_image, annotations = ds_test[i]

            image = Image.open(path)
            inputs = self.processor(image, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            w, h = image.size
            results = self.processor.post_process_object_detection(
                outputs, target_sizes=[(h, w)], threshold=0.3)

            detections = sv.Detections.from_transformers(results[0])

            targets.append(annotations)
            predictions.append(detections)

        # Calculate mAP
        mean_average_precision = sv.MeanAveragePrecision.from_detections(
            predictions=predictions,
            targets=targets,
        )

        print(f"mAP50_95: {mean_average_precision.map50_95:.2f}")
        print(f"mAP50: {mean_average_precision.map50:.2f}")
        print(f"mAP75: {mean_average_precision.map75:.2f}")

        # Calculate Confusion Matrix
        confusion_matrix = sv.ConfusionMatrix.from_detections(
            predictions=predictions,
            targets=targets,
            classes=ds_test.classes
        )

        _ = confusion_matrix.plot()

    def prepare_datasets(self, project_name, version_number):
        rf = Roboflow(api_key=self.roboflow_api_key)
        project = rf.workspace("subterraspace").project(project_name)
        version = project.version(version_number)
        dataset = version.download("coco")

        ds_train = sv.DetectionDataset.from_coco(
            images_directory_path=f"{dataset.location}/train",
            annotations_path=f"{dataset.location}/train/_annotations.coco.json",
        )
        ds_valid = sv.DetectionDataset.from_coco(
            images_directory_path=f"{dataset.location}/valid",
            annotations_path=f"{dataset.location}/valid/_annotations.coco.json",
        )
        ds_test = sv.DetectionDataset.from_coco(
            images_directory_path=f"{dataset.location}/test",
            annotations_path=f"{dataset.location}/test/_annotations.coco.json",
        )
        return ds_train, ds_valid, ds_test

    def fine_tune_model(self, ds_train, ds_valid, ds_test, epochs=100, batch_size=16):
        pytorch_dataset_train = PyTorchDetectionDataset(
            ds_train, self.processor)
        pytorch_dataset_valid = PyTorchDetectionDataset(
            ds_valid, self.processor)
        pytorch_dataset_test = PyTorchDetectionDataset(
            ds_test, self.processor)
        
        id2label = {id: label for id, label in enumerate(ds_train.classes)}
        label2id = {label: id for id, label in enumerate(ds_train.classes)}

        eval_compute_metrics_fn = MAPEvaluator(image_processor=self.processor, threshold=0.01, id2label=id2label)

        self.model = AutoModelForObjectDetection.from_pretrained(
            self.checkpoint,
            id2label=id2label,
            label2id=label2id,
            anchor_image_size=None,
            ignore_mismatched_sizes=True,
        )

        training_args = TrainingArguments(
            output_dir="finetuned-model",
            num_train_epochs=epochs,
            max_grad_norm=0.1,
            learning_rate=5e-5,
            warmup_steps=300,
            per_device_train_batch_size=batch_size,
            dataloader_num_workers=2,
            metric_for_best_model="eval_map",
            greater_is_better=True,
            load_best_model_at_end=True,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            remove_unused_columns=False,
            eval_do_concat_batches=False,
            resume_from_checkpoint=True,
        )

        checkpoint = "finetuned-model/checkpoint-126430"
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=pytorch_dataset_train,
            eval_dataset=pytorch_dataset_valid,
            tokenizer=self.processor,
            data_collator=self.collate_fn,
            compute_metrics=eval_compute_metrics_fn,
        )

        if checkpoint:
            trainer.train(resume_from_checkpoint=checkpoint)
        else:
            trainer.train()

        self.model.save_pretrained("models/rt-detr/")
        self.processor.save_pretrained("models/rt-detr/")
        return self.model

    @staticmethod
    def collate_fn(batch):
        data = {}
        data["pixel_values"] = torch.stack([x["pixel_values"] for x in batch])
        data["labels"] = [x["labels"] for x in batch]
        return data
