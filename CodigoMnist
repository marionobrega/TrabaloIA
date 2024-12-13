import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Configurações iniciais
#define o tamanho do lote, número de épocas e taxa de aprendizado
batch_size = 32  # Tamanho do lote para treinamento
epochs = 10  # Número de épocas
learning_rate = 0.001  # Taxa de aprendizado

# Transformações e data augmentation
transform = transforms.Compose([
    transforms.ToTensor(),  # Converte imagens para tensores PyTorch
    transforms.Normalize((0.5,), (0.5,))  # Normaliza os valores de pixel para o intervalo [-1, 1]
])

# Carregar o dataset MNIST
dataset_train = datasets.MNIST(root="./data", train=True, transform=transform, download=True)  # Dados de treinamento
dataset_test = datasets.MNIST(root="./data", train=False, transform=transform, download=True)  # Dados de teste

data_loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)  # Carrega os dados de treinamento em lotes
data_loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)  # Carrega os dados de teste em lotes

# Definição do modelo MLP
class MLP(nn.Module):  # Define a estrutura de uma rede neural feedforward
    def _init_(self):
        super(MLP, self)._init_()
        self.flatten = nn.Flatten()  # Achata a imagem de 28x28 para um vetor de 784 elementos
        self.fc1 = nn.Linear(28 * 28, 300)  # Camada oculta com 300 neurônios
        self.fc2 = nn.Linear(300, 100)  # Segunda camada oculta com 100 neurônios
        self.fc3 = nn.Linear(100, 10)  # Camada de saída com 10 neurônios (uma para cada classe)
        self.relu = nn.ReLU()  # Função de ativação ReLU
        self.softmax = nn.Softmax(dim=1)  # Função de ativação Softmax para saída

    def forward(self, x):
        x = self.flatten(x)  # Achata a entrada
        x = self.relu(self.fc1(x))  # Passa pela primeira camada e aplica ReLU
        x = self.relu(self.fc2(x))  # Passa pela segunda camada e aplica ReLU
        x = self.softmax(self.fc3(x))  # Passa pela camada de saída e aplica Softmax
        return x

model = MLP()  # Instancia o modelo

# Função de perda e otimizador
criterion = nn.CrossEntropyLoss()  # Função de perda para classificação (entropia cruzada)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Otimizador Adam com taxa de aprendizado definida

# Treinamento do modelo
train_loss = []  # Lista para armazenar a perda de treinamento
train_accuracy = []  # Lista para armazenar a acurácia de treinamento
val_loss = []  # Lista para armazenar a perda de validação
val_accuracy = []  # Lista para armazenar a acurácia de validação

for epoch in range(epochs):  # Loop sobre as épocas
    model.train()  # Coloca o modelo em modo de treinamento
    epoch_loss = 0  # Inicializa a perda da época
    correct = 0  # Inicializa o contador de previsões corretas
    total = 0  # Inicializa o total de amostras

    for images, labels in data_loader_train:  # Itera sobre os lotes de dados de treinamento
        outputs = model(images)  # Forward pass: gera previsões do modelo
        loss = criterion(outputs, labels)  # Calcula a perda

        optimizer.zero_grad()  # Zera os gradientes acumulados
        loss.backward()  # Calcula os gradientes
        optimizer.step()  # Atualiza os pesos

        epoch_loss += loss.item()  # Acumula a perda do lote
        _, predicted = outputs.max(1)  # Obtém as classes previstas
        total += labels.size(0)  # Conta o total de amostras
        correct += predicted.eq(labels).sum().item()  # Conta o número de previsões corretas

    train_loss.append(epoch_loss / len(data_loader_train))  # Calcula a perda média por lote
    train_accuracy.append(100. * correct / total)  # Calcula a acurácia de treinamento

    # Validação
    model.eval()  # Coloca o modelo em modo de avaliação
    val_epoch_loss = 0  # Inicializa a perda da época de validação
    val_correct = 0  # Inicializa o contador de previsões corretas na validação
    val_total = 0  # Inicializa o total de amostras na validação
    with torch.no_grad():  # Desativa o cálculo de gradientes durante a validação
        for images, labels in data_loader_test:  # Itera sobre os lotes de dados de validação
            outputs = model(images)  # Forward pass: gera previsões do modelo
            loss = criterion(outputs, labels)  # Calcula a perda
            val_epoch_loss += loss.item()  # Acumula a perda do lote
            _, predicted = outputs.max(1)  # Obtém as classes previstas
            val_total += labels.size(0)  # Conta o total de amostras
            val_correct += predicted.eq(labels).sum().item()  # Conta o número de previsões corretas

    val_loss.append(val_epoch_loss / len(data_loader_test))  # Calcula a perda média por lote na validação
    val_accuracy.append(100. * val_correct / val_total)  # Calcula a acurácia de validação

    # Exibe os resultados da época
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss / len(data_loader_train):.4f}, "
          f"Accuracy: {100. * correct / total:.2f}%, "
          f"Validation Loss: {val_epoch_loss / len(data_loader_test):.4f}, "
          f"Validation Accuracy: {100. * val_correct / val_total:.2f}%")

# Visualização dos resultados
plt.figure(figsize=(12, 5))

# Gráfico de perda
plt.subplot(1, 2, 1)
plt.plot(train_loss, label="Perda de Treinamento")
plt.plot(val_loss, label="Perda de Validação")
plt.title("Gráfico de Perda")
plt.xlabel("Épocas")
plt.ylabel("Perda")
plt.legend()

# Gráfico de acurácia
plt.subplot(1, 2, 2)
plt.plot(train_accuracy, label="Acurácia de Treinamento")
plt.plot(val_accuracy, label="Acurácia de Validação")
plt.title("Gráfico de Acurácia")
plt.xlabel("Épocas")
plt.ylabel("Acurácia (%)")
plt.legend()

plt.tight_layout()
plt.show()
