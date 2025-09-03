import pandas as pd
import matplotlib.pyplot as plt

def plot_metrics(path_to_csv):
    # Чтение данных из CSV файла
    data = pd.read_csv(str(path_to_csv)+'.csv')
    
    # Создание фигуры с 4 подграфиками (2x2)
    plt.figure(figsize=(15, 10))
    
    # График 1: Train Loss и Test Loss
    plt.subplot(2, 2, 1)
    if 'Train_Loss' in data.columns:
        plt.plot(data['Train_Loss'], label='Train Loss', color='blue')
        plt.title('Training and Test Loss')
    else:
        plt.title('Test Loss')
    plt.plot(data['Test_loss'], label='Test Loss', color='orange')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # График 2: Accuracy
    plt.subplot(2, 2, 2)
    plt.plot(data['Acceracy'], label='Accuracy', color='blue')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # График 3: Precision
    plt.subplot(2, 2, 3)
    plt.plot(data['Preciion'], label='Precision', color='blue')
    plt.title('Precision')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.legend()
    
    # График 4: Recall
    plt.subplot(2, 2, 4)
    plt.plot(data['Recall'], label='Recall', color='blue')
    plt.title('Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.legend()
    
    # Настройка отступов и отображение графиков
    plt.tight_layout()
    plt.savefig(str(path_to_csv)+'.pdf')


