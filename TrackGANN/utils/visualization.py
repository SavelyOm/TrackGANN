import pandas as pd
import matplotlib.pyplot as plt

def plot_metrics(path_to_csv):
    data = pd.read_csv(str(path_to_csv)+'.csv')
    
    
    plt.figure(figsize=(15, 10))
    
    
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
    
    
    plt.subplot(2, 2, 2)
    plt.plot(data['Acceracy'], label='Accuracy', color='blue')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    
    plt.subplot(2, 2, 3)
    plt.plot(data['Preciion'], label='Precision', color='blue')
    plt.title('Precision')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.legend()
    
   
    plt.subplot(2, 2, 4)
    plt.plot(data['Recall'], label='Recall', color='blue')
    plt.title('Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.legend()
    
   
    plt.tight_layout()
    plt.savefig(str(path_to_csv)+'.pdf')


