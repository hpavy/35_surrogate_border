import torch
import time
from utils import write_csv
from pathlib import Path


def train(
    nb_epoch,
    train_loss,
    test_loss,
    weight_data_init,
    weight_border_init,
    dynamic_weights,
    lr_weights,
    model,
    loss,
    optimizer,
    X_train,
    U_train,
    X_test_data,
    U_test_data,
    Re,
    time_start,
    f,
    folder_result,
    save_rate,
    batch_size,
    scheduler,
    X_border_train,
    X_border_test,
    U_border_train,
    U_border_test,
    mean_std,
    param_adim,
    nb_simu,
    u_border,
    v_border,
    p_border
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nb_it_tot = nb_epoch + len(train_loss["total"])
    print(
        f"--------------------------\nStarting at epoch: {len(train_loss['total'])}"
        + "\n--------------------------"
    )
    print(
        f"--------------------------\nStarting at epoch: {len(train_loss['total'])}\n------------"
        + "--------------",
        file=f,
    )

    if device == torch.device("cuda"):
        stream_data = torch.cuda.Stream()
        stream_border = torch.cuda.Stream()

    border_verify = [k for k, test_true in enumerate([u_border, v_border, p_border]) if test_true]

    weight_border = torch.tensor(weight_border_init, dtype=torch.float32, device=device)
    weight_data = torch.tensor(weight_data_init, dtype=torch.float32, device=device)

    nb_batches = 1000  #### A changer

    X_border_train = X_border_train.to(device)
    X_border_test = X_border_test.to(device).detach()
    U_border_train = U_border_train.to(device)
    U_border_test = U_border_test.to(device).detach()
    X_train = X_train.to(device)
    U_train = U_train.to(device)
    X_test_data = X_test_data.to(device).detach()
    U_test_data = U_test_data.to(device).detach()
    len_X_train_one = X_train.size(0) // nb_simu

    for epoch in range(len(train_loss["total"]), nb_it_tot):
        time_start_batch = time.time()
        total_batch = 0.0
        data_batch = 0.0
        border_batch = 0.0
        model.train()  # on dit qu'on va entrainer (on a le dropout)
        for nb_batch in range(nb_batches):
            with torch.cuda.stream(stream_data):
                X_train_batch = (
                    X_train[
                        (nb_batch % nb_simu)
                        * len_X_train_one:(nb_batch % nb_simu + 1)
                        * len_X_train_one
                    ]
                    .clone()
                    .requires_grad_()
                )
                U_train_batch = (
                    U_train[
                        (nb_batch % nb_simu)
                        * len_X_train_one:(nb_batch % nb_simu + 1)
                        * len_X_train_one
                    ]
                    .clone()
                    .requires_grad_()
                )
                # loss des points de data
                pred_data = model(X_train_batch)
                loss_data = loss(U_train_batch, pred_data)

            with torch.cuda.stream(stream_border):
                # loss du border
                pred_border = model(X_border_train)
                loss_border_cylinder = loss(pred_border[:, border_verify], U_border_train[:, border_verify])  # (MSE)

            torch.cuda.synchronize()

            loss_totale = weight_data * loss_data + weight_border * loss_border_cylinder

            # Backpropagation
            loss_totale.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                total_batch += loss_totale.item()
                data_batch += loss_data.item()
                border_batch += loss_border_cylinder.item()

        ######## A partir d'ici #######
        # Pour le test :
        model.eval()

        with torch.no_grad():
            # loss de la data
            test_data = model(X_test_data)
            loss_test_data = loss(U_test_data, test_data)  # (MSE)

            # loss des bords
            pred_border_test = model(X_border_test)
            loss_test_border = loss(pred_border_test[:, border_verify], U_border_test[:, border_verify])  # (MSE)

            # loss totale
            loss_test = weight_data * loss_test_data + weight_border * loss_test_border
        scheduler.step()

        # Weights
        with torch.no_grad():
            total_batch /= nb_batch
            data_batch /= nb_batch
            border_batch /= nb_batch
            if dynamic_weights:
                weight_data_hat = weight_data + lr_weights * data_batch
                weight_border_hat = weight_border + lr_weights * border_batch
                sum_weight = weight_data_hat + weight_border_hat
                weight_data = weight_data_hat / sum_weight
                weight_border = weight_border_hat / sum_weight
            test_loss["total"].append(loss_test.item())
            test_loss["data"].append(loss_test_data.item())
            test_loss["border"].append(loss_test_border.item())
            train_loss["total"].append(total_batch)
            train_loss["data"].append(data_batch)
            train_loss["border"].append(border_batch)

        print(f"---------------------\nEpoch {epoch+1}/{nb_it_tot} :")
        print(f"---------------------\nEpoch {epoch+1}/{nb_it_tot} :", file=f)
        print(
            f"Train : loss: {train_loss['total'][-1]:.3e}, data: {train_loss['data'][-1]:.3e}, border: {train_loss['border'][-1]:.3e}"
        )
        print(
            f"Train : loss: {train_loss['total'][-1]:.3e}, data: {train_loss['data'][-1]:.3e}, border: {train_loss['border'][-1]:.3e}",
            file=f,
        )
        print(
            f"Test  : loss: {test_loss['total'][-1]:.3e}, data: {test_loss['data'][-1]:.3e}, border: {test_loss['border'][-1]:.3e}"
        )
        print(
            f"Test  : loss: {test_loss['total'][-1]:.3e}, data: {test_loss['data'][-1]:.3e}, border: {test_loss['border'][-1]:.3e}",
            file=f,
        )
        print(
            f"Weights  : ------------, data: {weight_data.item():.1e},   border: {weight_border.item():.1e}"
        )
        print(
            f"Weights  : ------------, data: {weight_data.item():.1e},   border: {weight_border.item():.1e}",
            file=f,
        )

        print(f"time: {time.time()-time_start:.0f}s")
        print(f"time: {time.time()-time_start:.0f}s", file=f)

        print(f"time_epoch: {time.time()-time_start_batch:.0f}s")
        print(f"time: {time.time()-time_start_batch:.0f}s", file=f)

        if (epoch + 1) % save_rate == 0:
            with torch.no_grad():
                dossier_midle = Path(
                    folder_result + f"/epoch{len(train_loss['total'])}"
                )
                dossier_midle.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "weights": {
                            "weight_data": weight_data,
                            "weight_border": weight_border,
                        },
                    },
                    folder_result
                    + f"/epoch{len(train_loss['total'])}"
                    + "/model_weights.pth",
                )

                write_csv(
                    train_loss,
                    folder_result + f"/epoch{len(train_loss['total'])}",
                    file_name="/train_loss.csv",
                )
                write_csv(
                    test_loss,
                    folder_result + f"/epoch{len(train_loss['total'])}",
                    file_name="/test_loss.csv",
                )
