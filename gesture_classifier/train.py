import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from dataset import GestureDataset, GESTURE_CLASSES
from model import GestureClassifier


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for points, labels in loader:
        points = points.to(device)
        labels = labels.to(device)

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


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for points, labels in loader:
            points = points.to(device)
            labels = labels.to(device)

            outputs = model(points)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            pred = outputs.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(loader), correct / total


def run_experiment(alpha, seed, num_epochs=50, num_points=64, batch_size=32, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_dataset = GestureDataset(num_points=num_points, train=True)
    test_dataset = GestureDataset(num_points=num_points, train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = GestureClassifier(num_classes=len(GESTURE_CLASSES), alpha=alpha).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0

    for epoch in range(num_epochs):
        train_one_epoch(model, train_loader, optimizer, criterion, device)
        _, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        if test_acc > best_acc:
            best_acc = test_acc

    return best_acc


if __name__ == '__main__':
    alphas = [0.0, 0.5, 1.0, 2.0]
    seeds = [42, 123, 456, 789, 1234]
    num_runs = len(seeds)

    all_results = {alpha: [] for alpha in alphas}

    for alpha in alphas:
        print(f'\n{"="*50}')
        print(f'alpha={alpha}')
        print(f'{"="*50}')
        for i, seed in enumerate(seeds):
            acc = run_experiment(alpha=alpha, seed=seed, num_epochs=50)
            all_results[alpha].append(acc)
            print(f'  Run {i+1}/{num_runs} (seed={seed}): {acc:.4f}')

    print('\n' + '='*50)
    print('ABLATION STUDY RESULTS (mean ± std over 5 runs)')
    print('='*50)
    for alpha in alphas:
        accs = np.array(all_results[alpha])
        mean = accs.mean()
        std = accs.std()
        print(f'alpha={alpha}: {mean:.4f} ± {std:.4f} ({mean*100:.2f}% ± {std*100:.2f}%)')
