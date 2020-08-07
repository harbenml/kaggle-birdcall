from pathlib import Path

from fastai.vision import *
import torchvision

# set variables for reproducibility
torch.manual_seed(10)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(22)

saved_model_path = Path("bestmodel")
export_model_path = Path("models/export.pkl")

data = load_data(".", file="models/data_save.pkl", bs=64, num_workers=8)

os.environ["TORCH_HOME"] = "models/pretrained"

learn = create_cnn(data, torchvision.models.resnet34, metrics=accuracy, pretrained=True)
learn.load(saved_model_path)

print(learn.validate(metrics=[accuracy]))

learn.export(export_model_path)

learn_loaded = load_learner("models/")

print(learn_loaded.validate(dl=learn.data.valid_dl, metrics=[accuracy]))