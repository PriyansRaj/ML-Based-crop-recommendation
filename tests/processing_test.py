from src.preprocessing.processing import PreprocessingPipeline
if __name__ == "__main__":
    pipeline = PreprocessingPipeline()
    artifacts = pipeline.run_pipeline()
    print(artifacts)