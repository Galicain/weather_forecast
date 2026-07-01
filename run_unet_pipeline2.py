import os
import json
import math
import time
import zipfile
import argparse

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


############################################
# Dataset
############################################

class WeatherDataset(Dataset):
    """
    Dataset one-step:
    entrée  = état à t
    cible   = état à t+6h
    """
    def __init__(self, data):
        if data.ndim != 4:
            raise ValueError(f"Shape attendue (time, channels, H, W), reçu {data.shape}")

        if data.shape[0] < 2:
            raise ValueError("Pas assez de pas de temps pour construire le dataset.")

        self.X = data[:-1]
        self.y = data[1:]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        return x, y


############################################
# Modèle
############################################

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """
    U-Net léger, stable sur dimensions impaires grâce à interpolate.
    """
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        self.enc1 = DoubleConv(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(32, 64)

        self.dec1_conv = DoubleConv(64 + 32, 32)
        self.out_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)                 # (B,32,H,W)
        p1 = self.pool1(e1)               # (B,32,H/2,W/2)
        e2 = self.enc2(p1)                # (B,64,H/2,W/2)

        up = F.interpolate(
            e2,
            size=e1.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        cat = torch.cat([up, e1], dim=1)  # (B,96,H,W)
        d1 = self.dec1_conv(cat)          # (B,32,H,W)
        out = self.out_conv(d1)           # (B,1,H,W)

        return out


############################################
# Chargement ERA5
############################################

def load_one_file(file_path, variable_name="t"):
    print(f"Loading {file_path}")

    with xr.open_dataset(file_path) as ds:
        print("Variables disponibles:", list(ds.data_vars))
        print("Dimensions:", dict(ds.sizes))

        if variable_name not in ds:
            raise KeyError(
                f"Variable '{variable_name}' introuvable dans {file_path}. "
                f"Variables disponibles: {list(ds.data_vars)}"
            )

        da = ds[variable_name]

        if "valid_time" in da.dims and "time" not in da.dims:
            da = da.rename({"valid_time": "time"})

        if "pressure_level" in da.dims:
            da = da.isel(pressure_level=0)

        arr = da.values.astype(np.float32)

    if arr.ndim != 3:
        raise ValueError(
            f"Shape inattendue pour {file_path}: {arr.shape}. "
            "Attendu: (time, lat, lon)"
        )

    arr = arr[:, np.newaxis, :, :]  # (time, 1, H, W)
    return arr


def load_data(data_dir, variable_name="t"):
    files = sorted(
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith(".nc")
    )

    if not files:
        raise FileNotFoundError(f"Aucun fichier .nc trouvé dans {data_dir}")

    arrays = []
    for f in files:
        arrays.append(load_one_file(f, variable_name=variable_name))

    data = np.concatenate(arrays, axis=0)
    print("Shape finale des données:", data.shape)
    return data


############################################
# Normalisation
############################################

def normalize_data(data):
    mean = data.mean(axis=(0, 2, 3), keepdims=True)
    std = data.std(axis=(0, 2, 3), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)

    data_norm = (data - mean) / std

    stats = {
        "mean": mean.squeeze().tolist() if mean.size > 1 else float(mean.squeeze()),
        "std": std.squeeze().tolist() if std.size > 1 else float(std.squeeze()),
    }

    return data_norm.astype(np.float32), mean.astype(np.float32), std.astype(np.float32), stats


def denormalize_data(x, mean, std):
    return x * std + mean


############################################
# Split temporel
############################################

def split_data_timewise(data, train_ratio=0.7, val_ratio=0.15):
    n = data.shape[0]

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    if len(train_data) < 2 or len(val_data) < 2 or len(test_data) < 2:
        raise ValueError(
            f"Split invalide. train={len(train_data)}, val={len(val_data)}, test={len(test_data)}"
        )

    print(f"Split temporel -> train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
    return train_data, val_data, test_data


############################################
# Entraînement / validation
############################################

def evaluate_one_step(model, loader, device):
    model.eval()
    loss_fn = nn.MSELoss()

    total_loss = 0.0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            loss = loss_fn(pred, y)
            total_loss += loss.item()

    return total_loss / max(len(loader), 1)


def train_with_best_checkpoint(
    model,
    train_loader,
    val_loader,
    device,
    epochs,
    lr,
    checkpoint_path
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    history = {
        "train_loss": [],
        "val_loss": [],
    }

    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        train_loss = total_train_loss / max(len(train_loader), 1)
        val_loss = evaluate_one_step(model, val_loader, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(
            f"Epoch {epoch + 1}/{epochs} - "
            f"train_loss={train_loss:.6f} - val_loss(t+6)={val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"[best] checkpoint sauvegardé -> {checkpoint_path}")

    return history, best_val_loss


############################################
# Métriques
############################################

def compute_rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_metrics(y_true, y_pred):
    return {
        "rmse": compute_rmse(y_true, y_pred),
        "mae": compute_mae(y_true, y_pred),
    }


############################################
# Prédiction récursive multi-jours
############################################

def recursive_forecast_metrics(
    model,
    test_data_norm,
    mean,
    std,
    device,
    max_days=5,
    steps_per_day=4
):
    """
    Teste la prévision récursive:
    t -> t+6 -> t+12 -> ... sur plusieurs jours.

    test_data_norm shape: (time, 1, H, W)
    """
    model.eval()

    max_lead_steps = max_days * steps_per_day
    n_time = test_data_norm.shape[0]

    if n_time <= max_lead_steps:
        raise ValueError(
            f"Pas assez de données test ({n_time}) pour évaluer {max_days} jours."
        )

    lead_rmse = {step: [] for step in range(1, max_lead_steps + 1)}
    lead_mae = {step: [] for step in range(1, max_lead_steps + 1)}

    with torch.no_grad():
        for start_idx in range(0, n_time - max_lead_steps):
            current = test_data_norm[start_idx:start_idx + 1]   # (1,1,H,W)

            for lead_step in range(1, max_lead_steps + 1):
                x_t = torch.tensor(current, dtype=torch.float32, device=device)
                pred_norm = model(x_t).cpu().numpy()             # (1,1,H,W)

                true_norm = test_data_norm[start_idx + lead_step:start_idx + lead_step + 1]

                pred_denorm = denormalize_data(pred_norm, mean, std)
                true_denorm = denormalize_data(true_norm, mean, std)

                rmse = compute_rmse(true_denorm, pred_denorm)
                mae = compute_mae(true_denorm, pred_denorm)

                lead_rmse[lead_step].append(rmse)
                lead_mae[lead_step].append(mae)

                # récursion: la sortie devient l'entrée suivante
                current = pred_norm

    results = []
    for lead_step in range(1, max_lead_steps + 1):
        lead_hours = lead_step * 6
        results.append({
            "lead_step": lead_step,
            "lead_hours": lead_hours,
            "lead_days": lead_hours / 24.0,
            "rmse": float(np.mean(lead_rmse[lead_step])),
            "mae": float(np.mean(lead_mae[lead_step])),
            "n_samples": int(len(lead_rmse[lead_step])),
        })

    return results


############################################
# Sauvegardes graphiques
############################################

def save_loss_plot(history, out_dir):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history["train_loss"], label="train_loss")
    ax.plot(history["val_loss"], label="val_loss_t+6")
    ax.set_title("Training / Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.legend()

    path = os.path.join(out_dir, "loss_curve.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path



def save_prediction_example_plot(y_true, y_pred, out_dir, title_suffix=""):
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    def to_2d(x):
        x = np.asarray(x)

        # Cas (H, W)
        if x.ndim == 2:
            return x

        # Cas (1, H, W) ou (C, H, W) -> on prend le premier canal
        if x.ndim == 3:
            return x[0]

        # Cas (1, 1, H, W) ou (B, C, H, W) -> on prend batch 0, canal 0
        if x.ndim == 4:
            return x[0, 0]

        raise ValueError(f"Shape non supportée pour affichage: {x.shape}")

    y_true_2d = to_2d(y_true)
    y_pred_2d = to_2d(y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    im0 = axes[0].imshow(y_true_2d, aspect="auto")
    axes[0].set_title(f"True {title_suffix}")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(y_pred_2d, aspect="auto")
    axes[1].set_title(f"Pred {title_suffix}")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    path = os.path.join(out_dir, "prediction_example.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path



def save_recursive_metrics_plot(results, out_dir):
    lead_hours = [r["lead_hours"] for r in results]
    rmse = [r["rmse"] for r in results]
    mae = [r["mae"] for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(lead_hours, rmse, marker="o", label="RMSE")
    ax.plot(lead_hours, mae, marker="s", label="MAE")
    ax.set_title("Dégradation de la qualité en prévision récursive")
    ax.set_xlabel("Horizon de prévision (heures)")
    ax.set_ylabel("Erreur")
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = os.path.join(out_dir, "recursive_forecast_metrics.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def save_device_comparison_plot(timings, out_dir):
    """
    timings: dict du type {"cpu": 123.4, "cuda": 12.3}
    """
    devices = list(timings.keys())
    times = [timings[d] for d in devices]

    labels = {"cpu": "CPU", "cuda": "GPU"}
    display_labels = [labels.get(d, d) for d in devices]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(display_labels, times, color=["#4C72B0", "#DD8452"][:len(devices)])
    ax.set_title("Temps d'exécution total : CPU vs GPU")
    ax.set_ylabel("Temps (secondes)")
    ax.grid(True, axis="y", alpha=0.3)

    for bar, t in zip(bars, times):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{t:.1f}s",
            ha="center",
            va="bottom",
        )

    if len(times) == 2 and min(times) > 0:
        speedup = max(times) / min(times)
        faster = display_labels[times.index(min(times))]
        ax.text(
            0.5, 0.95,
            f"{faster} est {speedup:.1f}x plus rapide",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=9,
            style="italic",
        )

    path = os.path.join(out_dir, "device_comparison.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


############################################
# ZIP
############################################

def zip_results(project_dir):
    zip_path = f"{project_dir}_results.zip"

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(project_dir):
            for f in files:
                full_path = os.path.join(root, f)
                arcname = os.path.relpath(full_path, start=project_dir)
                zf.write(full_path, arcname=arcname)

    print(f"Results zipped: {zip_path}")
    return zip_path


############################################
# Main
############################################

def run_pipeline_on_device(
    device_str,
    args,
    train_dataset,
    val_dataset,
    test_dataset,
    test_data,
    mean,
    std,
    stats,
    project_dir,
):
    """
    Exécute tout le pipeline (train + eval + prévision récursive + sauvegardes)
    sur le device donné ("cpu" ou "cuda"), et renvoie le temps total écoulé.
    """
    os.makedirs(project_dir, exist_ok=True)
    device = torch.device(device_str)
    print(f"\n=== Exécution sur device: {device} ===")

    start_time = time.time()

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    model = UNet(in_channels=1, out_channels=1).to(device)

    best_model_path = os.path.join(project_dir, "best_model_tplus6.pt")

    print("Training one-step model (t -> t+6)")
    history, best_val_loss = train_with_best_checkpoint(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        checkpoint_path=best_model_path,
    )
    print(f"Best validation loss at t+6: {best_val_loss:.6f}")

    print("Reloading best model")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    print("Evaluating best model on one-step test")
    test_one_step_loss = evaluate_one_step(model, test_loader, device)
    print(f"One-step test loss (t+6): {test_one_step_loss:.6f}")

    x0, y0 = test_dataset[0]
    with torch.no_grad():
        pred0 = model(x0.unsqueeze(0).to(device)).cpu().squeeze(0).numpy()

    y0_np = y0.numpy()
    pred0_denorm = denormalize_data(pred0, mean, std)
    y0_denorm = denormalize_data(y0_np, mean, std)

    one_step_metrics = compute_metrics(y0_denorm, pred0_denorm)
    print("Example one-step metrics:", one_step_metrics)

    print("Running recursive multi-day forecast evaluation")
    recursive_results = recursive_forecast_metrics(
        model=model,
        test_data_norm=test_data,
        mean=mean,
        std=std,
        device=device,
        max_days=args.max_days,
        steps_per_day=4,
    )

    print("Recursive forecast metrics:")
    for r in recursive_results:
        print(
            f"lead={r['lead_hours']:>3}h | "
            f"RMSE={r['rmse']:.4f} | MAE={r['mae']:.4f} | n={r['n_samples']}"
        )

    elapsed = time.time() - start_time
    print(f"Temps total d'exécution sur {device_str}: {elapsed:.2f}s")

    print("Saving artifacts")
    torch.save(model.state_dict(), os.path.join(project_dir, "final_model_loaded_from_best.pt"))

    with open(os.path.join(project_dir, "normalization_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    with open(os.path.join(project_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    with open(os.path.join(project_dir, "one_step_test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_val_loss_tplus6": best_val_loss,
                "test_loss_tplus6": test_one_step_loss,
                "example_metrics": one_step_metrics,
                "elapsed_seconds": elapsed,
            },
            f,
            indent=2,
        )

    with open(os.path.join(project_dir, "recursive_forecast_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(recursive_results, f, indent=2)

    np.save(os.path.join(project_dir, "sample_target_tplus6_denorm.npy"), y0_denorm)
    np.save(os.path.join(project_dir, "sample_prediction_tplus6_denorm.npy"), pred0_denorm)

    save_loss_plot(history, project_dir)
    save_prediction_example_plot(
        y_true=y0_denorm,
        y_pred=pred0_denorm,
        out_dir=project_dir,
        title_suffix="t+6",
    )
    save_recursive_metrics_plot(recursive_results, project_dir)

    zip_results(project_dir)

    return elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--project_dir", default="./results")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--variable", default="t")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--max_days", type=int, default=5)
    parser.add_argument(
        "--compare_devices",
        action="store_true",
        default=True,
        help="Exécute le pipeline sur CPU et GPU (si dispo) et compare les temps.",
    )

    args = parser.parse_args()

    os.makedirs(args.project_dir, exist_ok=True)

    print("Loading ERA5 data")
    data = load_data(args.data_dir, variable_name=args.variable)

    print("Normalizing data")
    data_norm, mean, std, stats = normalize_data(data)

    print("Splitting data")
    train_data, val_data, test_data = split_data_timewise(
        data_norm,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    train_dataset = WeatherDataset(train_data)
    val_dataset = WeatherDataset(val_data)
    test_dataset = WeatherDataset(test_data)

    # Détermine les devices à tester
    devices_to_run = ["cpu"]
    if torch.cuda.is_available():
        devices_to_run.append("cuda")
    else:
        print("GPU non disponible : la comparaison portera uniquement sur CPU.")

    timings = {}

    for device_str in devices_to_run:
        device_project_dir = os.path.join(args.project_dir, device_str)
        elapsed = run_pipeline_on_device(
            device_str=device_str,
            args=args,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            test_data=test_data,
            mean=mean,
            std=std,
            stats=stats,
            project_dir=device_project_dir,
        )
        timings[device_str] = elapsed

    print("\n=== Résumé des temps d'exécution ===")
    for device_str, elapsed in timings.items():
        print(f"{device_str}: {elapsed:.2f}s")

    with open(os.path.join(args.project_dir, "device_timings.json"), "w", encoding="utf-8") as f:
        json.dump(timings, f, indent=2)

    if len(timings) >= 1:
        save_device_comparison_plot(timings, args.project_dir)

    print("Done")


if __name__ == "__main__":
    main()
