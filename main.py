#!/usr/bin/env python3
# Bitcoin Predictor - Main Script
# Prediksi harga Bitcoin menggunakan LSTM atau Random Forest

from bitcoinPredictor import BitcoinPredictor
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta

def main():
    print("=== BITCOIN PREDICTOR ===")
    print("Prediksi harga Bitcoin menggunakan Machine Learning\n")
    
    # Inisialisasi predictor
    predictor = BitcoinPredictor(sequenceLength=60, testSize=0.2)
    
    try:
        # 1. Ambil data Bitcoin
        data = predictor.fetchBitcoinData(period="2y")
        print(f"Rentang data: {data.index[0].date()} hingga {data.index[-1].date()}")
        print(f"Harga terakhir: ${data['Close'].iloc[-1]:,.2f}\n")
        
        # 2. Persiapkan data
        xTrain, xTest, yTrain, yTest = predictor.prepareData()
        
        # 3. Bangun model (LSTM jika TensorFlow tersedia, Random Forest jika tidak)
        model = predictor.buildLstmModel()
        
        # 4. Latih model
        history = predictor.trainModel(epochs=50, batchSize=32)
        
        # 5. Buat prediksi
        predictions, actual = predictor.makePredictions()
        
        # 6. Evaluasi model
        metrics = predictor.evaluateModel(predictions, actual)
        
        # 7. Visualisasi hasil
        predictor.plotResults(predictions, actual)
        
        # 8. Prediksi masa depan
        futureDays = 7
        futurePredictions = predictor.predictFuture(days=futureDays)
        
        print(f"\n=== PREDIKSI {futureDays} HARI KE DEPAN ===")
        currentDate = data.index[-1]
        for i, pred in enumerate(futurePredictions):
            futureDate = currentDate + timedelta(days=i+1)
            print(f"{futureDate.strftime('%Y-%m-%d')}: ${pred:,.2f}")
        
        # Plot prediksi masa depan
        plt.figure(figsize=(12, 6))
        
        # Plot harga historis (30 hari terakhir)
        recentData = data['Close'].tail(30)
        plt.plot(recentData.index, recentData.values, 
                label='Harga Historis', color='blue', linewidth=2)
        
        # Plot prediksi masa depan
        futureDates = [currentDate + timedelta(days=i+1) for i in range(futureDays)]
        plt.plot(futureDates, futurePredictions, 
                label='Prediksi', color='red', linewidth=2, marker='o')
        
        modelType = "LSTM" if not hasattr(predictor, 'xTestFlat') else "Random Forest"
        plt.title(f'Prediksi Harga Bitcoin - 7 Hari ke Depan ({modelType})')
        plt.xlabel('Tanggal')
        plt.ylabel('Harga (USD)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('bitcoinFuturePrediction.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Simpan hasil ke CSV
        resultsDF = pd.DataFrame({
            'Date': futureDates,
            'PredictedPrice': futurePredictions
        })
        resultsDF.to_csv('bitcoinPredictions.csv', index=False)
        print(f"\nHasil prediksi disimpan ke 'bitcoinPredictions.csv'")
        
        # Summary
        print("\n=== RINGKASAN ===")
        print(f"Model: {modelType}")
        print(f"Model RMSE: ${metrics['rmse']:,.2f}")
        print(f"Model MAPE: {metrics['mape']:.2f}%")
        print(f"Harga saat ini: ${data['Close'].iloc[-1]:,.2f}")
        print(f"Prediksi 1 hari: ${futurePredictions[0]:,.2f}")
        print(f"Prediksi 7 hari: ${futurePredictions[-1]:,.2f}")
        
        change1d = ((futurePredictions[0] - data['Close'].iloc[-1]) / data['Close'].iloc[-1]) * 100
        change7d = ((futurePredictions[-1] - data['Close'].iloc[-1]) / data['Close'].iloc[-1]) * 100
        
        print(f"Perubahan prediksi 1 hari: {change1d:+.2f}%")
        print(f"Perubahan prediksi 7 hari: {change7d:+.2f}%")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Pastikan Anda memiliki koneksi internet untuk mengambil data Bitcoin.")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()