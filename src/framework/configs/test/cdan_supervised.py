config = {
    # 显式指定模型名称和数据集名称（用于图片标题）
    "model_name": "CDAN_Supervised_best",          # 自定义名称
    "dataset_name": "ASCAD_variable_700",             # 自定义名称

    "feature_extractor_path": "D:/SCA_UDA/FRAMEWORK/results/models/cdan_supervised051101/best_feature_extractor.h5",
    "classifier_path": "D:/SCA_UDA/FRAMEWORK/results/models/cdan_supervised051101/best_classifier.h5",
    
    "ascad_database": "D:/SCA_UDA/data/processed/ascad-variable-700.h5",
    
    "num_traces": 10000,
    "target_byte": 2,
    "multilabel": 0,
    "simulated_key": 0,
    
    "save_file": "D:/SCA_UDA/framework/results/ge/supervised051104/ge_curve.png",
    "ge_data_path": "D:/SCA_UDA/framework/results/ge/supervised051104/ge_data.npy"
}