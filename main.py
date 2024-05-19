from utils import pipeline_infer, manual_infer

if __name__ == "__main__":
    model_dir = "/data/models/stable-diffusion-v1-4"
    prompt = "a photo, many cars, wide rodes, beautiful scenes"

    pipeline_infer(model_dir, prompt)

    manual_infer(model_dir, prompt)
