import os
import torch


def save(net, logger, hparams, epoch):
    # Create the path the checkpint will be saved at using the epoch number
    path = os.path.join(hparams["model_save_dir"], "epoch_" + str(epoch))

    # create a dictionary containing the logger info and model info that will be saved
    checkpoint = {
        "logs": logger.get_logs(),
        "params": net.state_dict(),
    }

    # save checkpoint
    torch.save(checkpoint, path)


def restore(net, logger, hparams):
    """Load back the model and logger from a given checkpoint
    epoch detailed in hps['restore_epoch'], if available"""
    path = os.path.join(
        hparams["model_save_dir"], "epoch_" + str(hparams["restore_epoch"])
    )

    if os.path.exists(path):
        try:
            checkpoint = torch.load(path)

            logger.restore_logs(checkpoint["logs"])
            net.load_state_dict(checkpoint["params"])

            if "lr" in checkpoint:
                hparams["lr"] = checkpoint["lr"]
                print("Using the Learning Rate that was found.")

            hparams["start_epoch"] = hparams["restore_epoch"]
            print("Net Restored!")

        except Exception as e:
            print("Restore Failed! Training from scratch.")
            print(e)
            hparams["start_epoch"] = 0

    else:
        print("Restore point unavailable. Training from scratch.")
        hparams["start_epoch"] = 0
