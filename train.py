import argparse
import torch
import wandb
import pandas as pd
from tqdm import tqdm

from project.link_prediction import LinkPredictor
from project.datasets.wikipedia import Wikipedia, WikipediaTest
from project.utils import get_logger

def main(args):
    logger = get_logger(__name__)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    nhids = args.nhids
    learning_rate = args.learning_rate
    max_epochs = args.max_epochs
    pred_save_path = args.pred_save_path
    experim_name = args.experim_name

    data_root = "./data" if args.wandb else "./default"

    config = {"nhids": nhids, "learning_rate": learning_rate, "max_epochs": max_epochs}
    
    logger.info(f"starting wandb experiment ({experim_name})")

    wandb.init(project="dsaa-23", entity="flursky", config=config, name=experim_name)

    dataset = Wikipedia(root=data_root)
    
    train_data = dataset[0]
    valid_data = dataset[1]
    test_data = dataset[2]

    train_data = train_data.to(device)
    valid_data = valid_data.to(device)
    test_data = test_data.to(device)

    nfeats = dataset.num_features

    model = LinkPredictor(nfeats=nfeats, nhids=nhids, learning_rate=learning_rate, device=device)

    model.train()

    logger.info(f"training will start for {max_epochs} epochs")

    for _ in tqdm(range(max_epochs), desc="Training"):
        train_loss = model.training_step(train_data)
        valid_loss, f1_score = model.validation_step(valid_data)

        metrics = {
            "train/loss": train_loss,
            "valid/loss": valid_loss,
            "valid/f1": f1_score
        }

        wandb.log(metrics)

    wandb.finish()

    logger.info(f"training finished")
    logger.info(f"last validation loss: {valid_loss}")
    logger.info(f"last validation f1 score: {f1_score}")

    # Testing the model
    model.eval()
    with torch.no_grad():
        test_loss, f1_socre = model.validation_step(test_data)
        logger.info(f"Loss: {test_loss}\t F1 Score: {f1_score * 100:.2f} %")

    # Submit to Kaggle
    logger.info("load Kaggle test dataset")

    test_dataset = WikipediaTest(root=data_root)
    submission_data = test_dataset[0]

    submission_data = submission_data.to(device)
    prediction_threshold = 0.5

    model.eval()

    with torch.no_grad():
        logits = model(submission_data.x, submission_data.edge_index)
        y_pred = torch.sigmoid(logits)

        y_pred = (y_pred > prediction_threshold).long().cpu()
        edge_id = submission_data.edge_id.cpu()

        np_data = torch.stack((edge_id, y_pred)).t().numpy()

        pd.DataFrame(data=np_data, columns=["id", "label"]) \
            .to_csv(pred_save_path, index=False)
        
        logger.info(f'submission file saved at {pred_save_path}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--nhids', default=16, type=int, help="Number of hidden units. Default: 16")
    parser.add_argument('--learning_rate', default=0.001, type=float, help="Learning rate. Default: 0.001")
    parser.add_argument('--max_epochs', default=2, type=int, help="Max training epochs. Default: 2")

    parser.add_argument('--pred_save_path', default='./data/predictions.csv', help="Path to save file. Default: ./data/predictions.csv")
    parser.add_argument('--experim_name', default='link-prediction', help="Wandb experiment name. Default: link-prediction")

    parser.add_argument('--wandb', default=False, action='store_true')

    args = parser.parse_args()

    main(args)

