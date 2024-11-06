using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;

namespace HeartDiseasePrediction
{
    // Class representing the heart disease dataset
    public class HeartDiseaseData
    {
        [LoadColumn(1)]
        public float Age { get; set; } // Age of the patient

        [LoadColumn(2)]
        public string Sex { get; set; } // Gender of the patient

        [LoadColumn(4)]
        public string Cp { get; set; } // Chest pain type

        [LoadColumn(5)]
        public float Trestbps { get; set; } // Resting blood pressure

        [LoadColumn(6)]
        public float Chol { get; set; } // Serum cholesterol in mg/dl

        [LoadColumn(7)]
        public bool Fbs { get; set; } // Fasting blood sugar > 120 mg/dl

        [LoadColumn(8)]
        public string Restecg { get; set; } // Resting electrocardiographic results

        [LoadColumn(9)]
        public float Thalch { get; set; } // Maximum heart rate achieved

        [LoadColumn(10)]
        public bool Exang { get; set; } // Exercise induced angina

        [LoadColumn(11)]
        public float Oldpeak { get; set; } // ST depression induced by exercise

        [LoadColumn(12)]
        public string Slope { get; set; } // Slope of the peak exercise ST segment

        [LoadColumn(13)]
        public float Ca { get; set; } // Number of major vessels (0-3)

        [LoadColumn(14)]
        public string Thal { get; set; } // Thalassemia

        [LoadColumn(15)]
        public float Num { get; set; } // Diagnosis (0-1)
    }

    class Program
    {
        static void Main(string[] args)
        {
            // Create a new MLContext for ML.NET operations
            var mlContext = new MLContext();

            // Load the dataset from the specified path
            string dataPath = "C:/Users/stere/source/repos/Lab1/heart_disease_uci.csv";
            IDataView data = mlContext.Data.LoadFromTextFile<HeartDiseaseData>(dataPath, separatorChar: ',', hasHeader: true);

            // Split the data into training and testing sets (80/20 split)
            var trainTestSplit = mlContext.Data.TrainTestSplit(data, testFraction: 0.2);
            var trainingData = trainTestSplit.TrainSet; // Training data
            var testData = trainTestSplit.TestSet; // Testing data

            // Define data processing and training pipeline
            var dataProcessPipeline = mlContext.Transforms.Conversion.ConvertType("Fbs", outputKind: DataKind.Single)
                .Append(mlContext.Transforms.Conversion.ConvertType("Exang", outputKind: DataKind.Single))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("Sex"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("Cp"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("Restecg"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("Slope"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("Thal"))
                .Append(mlContext.Transforms.Concatenate("Features", "Age", "Sex", "Cp", "Trestbps", "Chol", "Fbs", "Restecg", "Thalch", "Exang", "Oldpeak", "Slope", "Ca"));

            // Specify the regression trainer with the label column correctly
            var trainer = mlContext.Regression.Trainers.Sdca(
                labelColumnName: "Num", // Column containing the label
                maximumNumberOfIterations: 2000); // Set maximum number of iterations for training

            // Combine the data processing and trainer into a training pipeline
            var trainingPipeline = dataProcessPipeline.Append(trainer);

            // Train the model using the training data
            var model = trainingPipeline.Fit(trainingData);

            // Evaluate the trained model on the test data
            var predictions = model.Transform(testData);

            // Evaluate the model performance using the specified label column
            var metrics = mlContext.Regression.Evaluate(predictions, labelColumnName: "Num", scoreColumnName: "Score");

            // Print model evaluation metrics to the console
            Console.WriteLine("Model Evaluation Metrics:");
            Console.WriteLine($"Mean Absolute Error (MAE): {metrics.MeanAbsoluteError}");
            Console.WriteLine($"Root Mean Squared Error (RMSE): {metrics.RootMeanSquaredError}");
            Console.WriteLine($"R-squared (R2): {metrics.RSquared}");

            // Save the trained model to a file
            using (var fileStream = new FileStream("model.zip", FileMode.Create, FileAccess.Write, FileShare.Write))
            {
                mlContext.Model.Save(model, trainingData.Schema, fileStream);
            }

            // Load the saved model from the file for future predictions
            ITransformer loadedModel;
            using (var fileStream = new FileStream("model.zip", FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                loadedModel = mlContext.Model.Load(fileStream, out var modelInputSchema);
            }

            // Create a prediction engine from the loaded model
            var predictionEngine = mlContext.Model.CreatePredictionEngine<HeartDiseaseData, HeartDiseasePrediction>(loadedModel);

            // Sample input data for making a prediction
            var newData = new HeartDiseaseData
            {
                Age = 45,
                Sex = "Female",
                Cp = "asymptomatic",
                Trestbps = 130,
                Chol = 250,
                Fbs = false,
                Restecg = "normal",
                Thalch = 150,
                Exang = true,
                Oldpeak = 1.5f,
                Slope = "flat",
                Ca = 1,
                Thal = "normal"
            };

            // Use the prediction engine to predict the diagnosis based on new data
            var prediction = predictionEngine.Predict(newData);
            Console.WriteLine($"\nPrediction for new data: Age={newData.Age}, Sex={newData.Sex}, Cp={newData.Cp}, Trestbps={newData.Trestbps}, Chol={newData.Chol}, Fbs={newData.Fbs}, Restecg={newData.Restecg}, Thalch={newData.Thalch}, Exang={newData.Exang}, Oldpeak={newData.Oldpeak}, Slope={newData.Slope}, Ca={newData.Ca}, Thal={newData.Thal}");
            Console.WriteLine($"Predicted Diagnosis: {prediction.Num}");

            // Check if the model performance is acceptable based on R-squared value
            if (metrics.RSquared >= 0.7)
            {
                Console.WriteLine("The model performs well and can be used for prediction on this type of data.");
            }
            else
            {
                Console.WriteLine("The model may not be accurate enough for reliable predictions. Consider improving the model.");
            }

            Console.WriteLine("Press Enter to exit...");
            Console.ReadLine();
        }
    }

    // Class for predicting the output of heart disease diagnosis
    public class HeartDiseasePrediction
    {
        [ColumnName("Score")]
        public float Num { get; set; } // Predicted diagnosis
    }
}
