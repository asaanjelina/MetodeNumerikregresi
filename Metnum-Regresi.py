import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

# Membaca data dari file CSV
file_path = r"C:\Users\Public\Aplikasi_Regresi\student_performance.csv"
data = pd.read_csv(file_path)

# Mengambil kolom yang diperlukan
NL = data['Sample Question Papers Practiced'].values.reshape(-1, 1)
NT = data['Performance Index'].values

# Melihat distribusi data
plt.figure(figsize=(7, 6))
plt.scatter(NL, NT, color='blue', label='Data')
plt.title('Scatter Plot of Data')
plt.xlabel('Sample Question Papers Practiced')
plt.ylabel('Performance Index')
plt.legend()
plt.show()

# Menghitung korelasi
correlation, _ = pearsonr(NL.flatten(), NT)
print(f'Correlation between NL and NT: {correlation}')

# Model Linear (Metode 1)
linear_model = LinearRegression()
linear_model.fit(NL, NT)
NT_pred_linear = linear_model.predict(NL)

# Model Eksponensial (Metode 3)
def exp_model(x, a, b):
    return a * np.exp(b * x)

popt, _ = curve_fit(exp_model, NL.flatten(), NT)
NT_pred_exp = exp_model(NL, *popt)

# Menghitung galat RMS
rms_linear = np.sqrt(mean_squared_error(NT, NT_pred_linear))
rms_exp = np.sqrt(mean_squared_error(NT, NT_pred_exp))

# Plot grafik titik data dan hasil regresinya
plt.figure(figsize=(14, 6))

# Plot untuk regresi linear
plt.subplot(1, 2, 1)
plt.scatter(NL, NT, color='blue', label='Data')
plt.plot(NL, NT_pred_linear, color='red', label='Linear Fit')
plt.title('Linear Regression')
plt.xlabel('Sample Question Papers Practiced')
plt.ylabel('Performance Index')
plt.legend()

# Plot untuk regresi eksponensial
plt.subplot(1, 2, 2)
plt.scatter(NL, NT, color='blue', label='Data')
plt.plot(NL, NT_pred_exp, color='green', label='Exponential Fit')
plt.title('Exponential Regression')
plt.xlabel('Sample Question Papers Practiced')
plt.ylabel('Performance Index')
plt.legend()

plt.tight_layout()
plt.show()

# Print RMS error
print(f'RMS Error for Linear Model: {rms_linear}')
print(f'RMS Error for Exponential Model: {rms_exp}')


