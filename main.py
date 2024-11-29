from methods.model_trainer import ModelTrainer
import os

def main():
    # Set API Key from environment or direct input
    roboflow_api_key = os.getenv('ROBOFLOW_API_KEY') or input("Enter your Roboflow API Key: ")

    # Instantiate ModelTrainer with parameters
    trainer = ModelTrainer(checkpoint="PekingU/rtdetr_r50vd_coco_o365", roboflow_api_key=roboflow_api_key)

    # Prepare datasets from Roboflow
    ds_train, ds_valid, ds_test = trainer.prepare_datasets("wild-pig-mtqln", version_number=3)

    # Fine-tune the model
    model = trainer.fine_tune_model(ds_train, ds_valid, ds_test)

if __name__ == "__main__":
    main()
