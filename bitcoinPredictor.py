import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

class BitcoinPredictor:
    def __init__(self, sequenceLength=60, testSize=0.2):
        self.sequenceLength = sequenceLength
        self.testSize = testSize
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.data = None
        
    def fetchBitcoinData(self, period="2y"):
        # Mengambil data Bitcoin dari Yahoo Finance
        print("Mengambil data Bitcoin...")
        ticker = yf.Ticker("BTC-USD")
        self.data = ticker.history(period=period)
        
        # Menambahkan fitur teknikal
        self.data['ma7'] = self.data['Close'].rolling(window=7).mean()
        self.data['ma21'] = self.data['Close'].rolling(window=21).mean()
        self.data['rsi'] = self.calculateRsi(self.data['Close'])
        self.data['volatility'] = self.data['Close'].rolling(window=10).std()
        
        # Menghapus nilai NaN
        self.data = self.data.dropna()
        
        print(f"Data berhasil diambil: {len(self.data)} hari")
        return self.data
    
    def calculateRsi(self, prices, window=14):
        # Menghitung Relative Strength Index (RSI)
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def prepareData(self, features=['Close', 'Volume', 'ma7', 'ma21', 'rsi', 'volatility']):
        # Mempersiapkan data untuk training
        print("Mempersiapkan data...")
        
        # Pilih fitur yang akan digunakan
        featureData = self.data[features].values
        
        # Normalisasi data
        scaledData = self.scaler.fit_transform(featureData)
        
        # Buat sequences
        X, y = [], []
        for i in range(self.sequenceLength, len(scaledData)):
            X.append(scaledData[i-self.sequenceLength:i])
            y.append(scaledData[i, 0])  # Prediksi harga Close
        
        X, y = np.array(X), np.array(y)
        
        # Split data
        splitIndex = int(len(X) * (1 - self.testSize))
        
        self.xTrain = X[:splitIndex]
        self.xTest = X[splitIndex:]
        self.yTrain = y[:splitIndex]
        self.yTest = y[splitIndex:]
        
        print(f"Data training: {self.xTrain.shape}")
        print(f"Data testing: {self.xTest.shape}")
        
        return self.xTrain, self.xTest, self.yTrain, self.yTest
    
    def buildLstmModel(self):
        # Membangun model LSTM (memerlukan TensorFlow)
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            from tensorflow.keras.optimizers import Adam
            
            print("Membangun model LSTM...")
            
            self.model = Sequential()
            
            # Layer LSTM pertama
            self.model.add(LSTM(units=50, return_sequences=True, 
                               input_shape=(self.xTrain.shape[1], self.xTrain.shape[2])))
            self.model.add(Dropout(0.2))
            
            # Layer LSTM kedua
            self.model.add(LSTM(units=50, return_sequences=False))
            self.model.add(Dropout(0.2))
            
            # Layer Dense
            self.model.add(Dense(units=25))
            self.model.add(Dense(units=1))
            
            # Compile model
            optimizer = Adam(learning_rate=0.001)
            self.model.compile(optimizer=optimizer, loss='mean_squared_error')
            
            print("Model LSTM berhasil dibangun!")
            return self.model
            
        except ImportError:
            print("TensorFlow tidak tersedia. Menggunakan Random Forest sebagai alternatif...")
            return self.buildRfModel()
    
    def buildRfModel(self):
        # Membangun model Random Forest sebagai alternatif
        from sklearn.ensemble import RandomForestRegressor
        
        print("Membangun model Random Forest...")
        
        # Reshape data untuk Random Forest (flatten sequences)
        self.xTrainFlat = self.xTrain.reshape(self.xTrain.shape[0], -1)
        self.xTestFlat = self.xTest.reshape(self.xTest.shape[0], -1)
        
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        print("Model Random Forest berhasil dibangun!")
        return self.model
    
    def trainModel(self, epochs=50, batchSize=32):
        # Melatih model
        print("Memulai training model...")
        
        if hasattr(self.model, 'fit'):
            if 'tensorflow' in str(type(self.model)):
                # Training LSTM
                history = self.model.fit(
                    self.xTrain, self.yTrain,
                    epochs=epochs,
                    batch_size=batchSize,
                    validation_split=0.1,
                    verbose=1
                )
                return history
            else:
                # Training Random Forest
                self.model.fit(self.xTrainFlat, self.yTrain)
                print("Training Random Forest selesai!")
                return None
    
    def makePredictions(self):
        # Membuat prediksi
        print("Membuat prediksi...")
        
        if hasattr(self, 'xTestFlat'):
            # Random Forest prediction
            predictions = self.model.predict(self.xTestFlat)
        else:
            # LSTM prediction
            predictions = self.model.predict(self.xTest)
        
        # Denormalisasi prediksi
        dummyArray = np.zeros((len(predictions), self.scaler.n_features_in_))
        dummyArray[:, 0] = predictions.flatten()
        predictionsDenorm = self.scaler.inverse_transform(dummyArray)[:, 0]
        
        # Denormalisasi actual values
        dummyArrayActual = np.zeros((len(self.yTest), self.scaler.n_features_in_))
        dummyArrayActual[:, 0] = self.yTest
        actualDenorm = self.scaler.inverse_transform(dummyArrayActual)[:, 0]
        
        return predictionsDenorm, actualDenorm
    
    def evaluateModel(self, predictions, actual):
        # Evaluasi performa model
        mse = mean_squared_error(actual, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual, predictions)
        mape = np.mean(np.abs((actual - predictions) / actual)) * 100
        
        print("\n=== EVALUASI MODEL ===")
        print(f"Mean Squared Error (MSE): ${mse:,.2f}")
        print(f"Root Mean Squared Error (RMSE): ${rmse:,.2f}")
        print(f"Mean Absolute Error (MAE): ${mae:,.2f}")
        print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
        
        return {'mse': mse, 'rmse': rmse, 'mae': mae, 'mape': mape}
    
    def plotResults(self, predictions, actual):
        # Visualisasi hasil prediksi
        plt.figure(figsize=(15, 6))
        
        # Plot perbandingan prediksi vs aktual
        plt.subplot(1, 2, 1)
        plt.plot(actual, label='Harga Aktual', color='blue', alpha=0.7)
        plt.plot(predictions, label='Prediksi', color='red', alpha=0.7)
        plt.title('Prediksi vs Harga Aktual Bitcoin')
        plt.xlabel('Hari')
        plt.ylabel('Harga (USD)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Scatter plot
        plt.subplot(1, 2, 2)
        plt.scatter(actual, predictions, alpha=0.6)
        plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', lw=2)
        plt.xlabel('Harga Aktual (USD)')
        plt.ylabel('Prediksi (USD)')
        plt.title('Scatter Plot: Prediksi vs Aktual')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('bitcoinPredictionResults.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Plot disimpan sebagai 'bitcoinPredictionResults.png'")
    
    def predictFuture(self, days=7):
        # Prediksi harga Bitcoin untuk beberapa hari ke depan
        print(f"Memprediksi harga Bitcoin untuk {days} hari ke depan...")
        
        # Ambil data terakhir
        lastSequence = self.data[['Close', 'Volume', 'ma7', 'ma21', 'rsi', 'volatility']].tail(self.sequenceLength).values
        lastSequenceScaled = self.scaler.transform(lastSequence)
        
        futurePredictions = []
        currentSequence = lastSequenceScaled.copy()
        
        for _ in range(days):
            if hasattr(self, 'xTestFlat'):
                # Random Forest prediction
                currentInput = currentSequence.flatten().reshape(1, -1)
                nextPred = self.model.predict(currentInput)
            else:
                # LSTM prediction
                currentInput = currentSequence.reshape(1, self.sequenceLength, -1)
                nextPred = self.model.predict(currentInput, verbose=0)
            
            # Denormalisasi prediksi
            dummyArray = np.zeros((1, self.scaler.n_features_in_))
            dummyArray[0, 0] = nextPred[0] if hasattr(self, 'xTestFlat') else nextPred[0, 0]
            nextPredDenorm = self.scaler.inverse_transform(dummyArray)[0, 0]
            
            futurePredictions.append(nextPredDenorm)
            
            # Update sequence untuk prediksi berikutnya
            newRow = currentSequence[-1].copy()
            newRow[0] = nextPred[0] if hasattr(self, 'xTestFlat') else nextPred[0, 0]
            
            # Shift sequence
            currentSequence = np.vstack([currentSequence[1:], newRow])
        
        return futurePredictions