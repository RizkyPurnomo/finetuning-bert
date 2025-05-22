import os
import pandas as pd
import helpers
import preprocessing
import finetuning

def main():
    # Load dataset
    datasets = helpers.get_dataset_names()

    for dataset_name in datasets:
        print(f"Processing {dataset_name} dataset...")

        # Step 1: Preprocessing
        print("Applying preprocessing...")
        best_preprocess_method = preprocessing.evaluate_preprocessing_methods(dataset_name)
        print(f"Best Preprocess Method for {dataset_name} dataset is ", best_preprocess_method)

        # Step 2: Fine-Tuning Step 1
        print("Step 1: Domain-specific model fine-tuning...")
        best_model, _ = fine_tuning_step_1(dataset)
        
        # Step 3: Fine-Tuning Step 2
        print("Step 2: Hyperparameter tuning...")
        best_model, _ = fine_tuning_step_2(best_model, dataset)
        
        # Step 4: Fine-Tuning Step 3
        print("Step 3: Feature-based approach...")
        best_model = fine_tuning_step_3(best_model, dataset)
        
        # Step 5: Fine-Tuning Step 4
        print("Step 4: Classifier testing...")
        fine_tuning_step_4(best_model, dataset, params={})

    print("Pipeline completed.")


if __name__ == "__main__":
    main()