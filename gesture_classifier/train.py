import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from dataset import GestureDataset, GESTURE_CLASSES
from model import GestureClassifier


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for points, labels in loader:
        points, labels = points.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(points)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred = outputs.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    return total_loss / len(loader), correct / total


def evaluate(model, loader, criterion, device, return_predictions=False):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for points, labels in loader:
            points, labels = points.to(device), labels.to(device)
            outputs = model(points)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            pred = outputs.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
            if return_predictions:
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
    acc = correct / total
    avg_loss = total_loss / len(loader)
    if return_predictions:
        return avg_loss, acc, np.array(all_preds), np.array(all_labels)
    return avg_loss, acc


def compute_metrics(preds, labels, class_names):
    num_classes = len(class_names)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for p, l in zip(preds, labels):
        cm[l][p] += 1

    metrics = {}
    for i, name in enumerate(class_names):
        tp = cm[i][i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        metrics[name] = {'precision': precision, 'recall': recall, 'f1': f1}
    return cm, metrics


def print_metrics(cm, metrics, class_names):
    print('\n  Confusion Matrix:')
    header = '         ' + '  '.join(f'{c:>7s}' for c in class_names)
    print(header)
    for i, name in enumerate(class_names):
        row = '  '.join(f'{cm[i][j]:>7d}' for j in range(len(class_names)))
        print(f'    {name:>7s}  {row}')

    print(f'\n  {"Class":<10s} {"Precision":>10s} {"Recall":>10s} {"F1":>10s}')
    print(f'  {"-"*40}')
    for name in class_names:
        m = metrics[name]
        print(f'  {name:<10s} {m["precision"]:>10.4f} {m["recall"]:>10.4f} {m["f1"]:>10.4f}')

    macro_p = np.mean([metrics[n]['precision'] for n in class_names])
    macro_r = np.mean([metrics[n]['recall'] for n in class_names])
    macro_f1 = np.mean([metrics[n]['f1'] for n in class_names])
    print(f'  {"-"*40}')
    print(f'  {"Macro Avg":<10s} {macro_p:>10.4f} {macro_r:>10.4f} {macro_f1:>10.4f}')


def run_experiment(alpha, seed, data_root, test_users=None, num_epochs=50,
                   num_points=64, batch_size=32, lr=0.001, verbose=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_dataset = GestureDataset(data_root=data_root, num_points=num_points, train=True, test_users=test_users)
    test_dataset = GestureDataset(data_root=data_root, num_points=num_points, train=False, test_users=test_users)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = GestureClassifier(num_classes=len(GESTURE_CLASSES), alpha=alpha).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    train_losses, test_accs = [], []

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        _, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()
        train_losses.append(train_loss)
        test_accs.append(test_acc)

        if verbose and (epoch + 1) % 10 == 0:
            print(f'    Epoch {epoch+1}/{num_epochs} — Loss: {train_loss:.4f}, '
                  f'Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

        if test_acc > best_acc:
            best_acc = test_acc

    _, final_acc, preds, labels = evaluate(model, test_loader, criterion, device, return_predictions=True)
    cm, metrics = compute_metrics(preds, labels, GESTURE_CLASSES)

    return best_acc, final_acc, cm, metrics, train_losses, test_accs


def plot_curves(train_losses, test_accs, title, save_path):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(train_losses, color='#e74c3c', linewidth=1.5)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Training Loss')
        ax1.set_title('Training Loss')
        ax1.grid(True, alpha=0.3)

        ax2.plot([a * 100 for a in test_accs], color='#2ecc71', linewidth=1.5)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Test Accuracy (%)')
        ax2.set_title('Test Accuracy')
        ax2.grid(True, alpha=0.3)

        fig.suptitle(title, fontsize=13, fontweight='bold')
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f'  Saved curves to {save_path}')
    except ImportError:
        print('  (matplotlib not available)')


if __name__ == '__main__':
    DATA_ROOT = os.environ.get('GESTURE_DATA_ROOT', 'data/')

    alphas = [0.0, 0.5, 1.0, 2.0]
    seeds = [42, 123, 456, 789, 1234]

    all_results = {a: [] for a in alphas}
    best_alpha_acc, best_alpha_value = -1, None

    print("Ablation Study (Subject-Dependent)")
    for alpha in alphas:
        print(f'\n{"="*50}\nalpha={alpha}\n{"="*50}')
        for i, seed in enumerate(seeds):
            best_acc, final_acc, cm, metrics, t_losses, t_accs = run_experiment(
                alpha=alpha, seed=seed, data_root=DATA_ROOT, num_epochs=50
            )
            all_results[alpha].append(best_acc)
            print(f'  Run {i+1}/{len(seeds)} (seed={seed}): Best={best_acc:.4f}, Final={final_acc:.4f}')

        print_metrics(cm, metrics, GESTURE_CLASSES)
        plot_curves(t_losses, t_accs, f'alpha={alpha}', f'gesture_classifier/curves_alpha_{alpha}.png')

        mean_acc = np.mean(all_results[alpha])
        if mean_acc > best_alpha_acc:
            best_alpha_acc = mean_acc
            best_alpha_value = alpha

    print('\n' + '='*50)
    print('ABLATION RESULTS (mean +/- std over 5 runs)')
    print('='*50)
    for alpha in alphas:
        accs = np.array(all_results[alpha])
        print(f'alpha={alpha}: {accs.mean():.4f} +/- {accs.std():.4f} '
              f'({accs.mean()*100:.2f}% +/- {accs.std()*100:.2f}%)')
    print(f'\nBest alpha: {best_alpha_value} ({best_alpha_acc*100:.2f}%)')

    # --- LOUO-CV ---
    print('\n' + '='*50)
    print(f'3-Fold LOUO-CV (alpha={best_alpha_value})')
    print('='*50)

    louo_users = [['001'], ['002'], ['004']]
    louo_results = []

    for i, test_users in enumerate(louo_users):
        print(f'\nFold {i+1}/3: test user {test_users}')
        best_acc, final_acc, cm, metrics, t_losses, t_accs = run_experiment(
            alpha=best_alpha_value, seed=42, data_root=DATA_ROOT,
            test_users=test_users, num_epochs=50, verbose=True
        )
        louo_results.append(best_acc)
        print(f'  Best: {best_acc:.4f}, Final: {final_acc:.4f}')
        print_metrics(cm, metrics, GESTURE_CLASSES)
        plot_curves(t_losses, t_accs, f'LOUO Fold {i+1}: user {test_users[0]}',
                    f'gesture_classifier/curves_louo_fold{i+1}.png')

    louo_accs = np.array(louo_results)
    print('\n' + '='*50)
    print('LOUO-CV RESULTS')
    print('='*50)
    print(f'alpha={best_alpha_value}: {louo_accs.mean():.4f} +/- {louo_accs.std():.4f} '
          f'({louo_accs.mean()*100:.2f}% +/- {louo_accs.std()*100:.2f}%)')
    for i, (users, acc) in enumerate(zip(louo_users, louo_results)):
        print(f'  Fold {i+1} (user {users[0]}): {acc:.4f} ({acc*100:.2f}%)')
