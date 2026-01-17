using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using System;
using System.Linq;

namespace BikeSharingPrediction
{
    class Program
    {
        // Путь к файлу данных
        private static string _dataPath = "bike_sharing.csv";

        // Класс для входных данных с boolean меткой
        public class BikeRentalData
        {
            [LoadColumn(0)]
            public float Season { get; set; }

            [LoadColumn(1)]
            public float Month { get; set; }

            [LoadColumn(2)]
            public float Hour { get; set; }

            [LoadColumn(3)]
            public float Holiday { get; set; }

            [LoadColumn(4)]
            public float Weekday { get; set; }

            [LoadColumn(5)]
            public float WorkingDay { get; set; }

            [LoadColumn(6)]
            public float WeatherCondition { get; set; }

            [LoadColumn(7)]
            public float Temperature { get; set; }

            [LoadColumn(8)]
            public float Humidity { get; set; }

            [LoadColumn(9)]
            public float Windspeed { get; set; }

            // Используем boolean напрямую
            [LoadColumn(10)]
            public bool RentalType { get; set; }
        }

        // Класс для результата предсказания
        public class RentalTypePrediction
        {
            [ColumnName("PredictedLabel")]
            public bool PredictedRentalType { get; set; }

            public float Probability { get; set; }

            public float Score { get; set; }
        }

        static void Main(string[] args)
        {
            Console.WriteLine("Предсказание типа аренды велосипеда с использованием ML.NET");

            // 1. Создаем контекст ML.NET
            var mlContext = new MLContext(seed: 0);

            try
            {
                // 2. Загружаем данные из CSV-файла
                Console.WriteLine("Загрузка данных...");

                // Загружаем данные из CSV файла
                var data = mlContext.Data.LoadFromTextFile<BikeRentalData>(_dataPath, hasHeader: true, separatorChar: ',');

                // Проверяем количество записей каждого типа
                var enumerableData = mlContext.Data.CreateEnumerable<BikeRentalData>(data, reuseRowObject: false);
                var countFalse = enumerableData.Count(x => x.RentalType == false);
                var countTrue = enumerableData.Count(x => x.RentalType == true);
                Console.WriteLine($"Распределение классов:");
                Console.WriteLine($"  RentalType = False: {countFalse} записей");
                Console.WriteLine($"  RentalType = True:  {countTrue} записей");

                // Проверяем, что у нас есть оба класса
                if (countFalse == 0 || countTrue == 0)
                {
                    Console.WriteLine("Внимание: В данных не хватает одного из классов!");
                }

                // 3. Разделяем на обучающую и тестовую выборки
                Console.WriteLine("Разделение данных...");
                var splitData = mlContext.Data.TrainTestSplit(data, testFraction: 0.1);

                // 4. Создаем пайплайн обработки признаков
                Console.WriteLine("Создание пайплайна...");

                var pipeline = mlContext.Transforms.Categorical.OneHotEncoding("Season")
                    .Append(mlContext.Transforms.Categorical.OneHotEncoding("WeatherCondition"))
                    .Append(mlContext.Transforms.NormalizeMinMax("Temperature"))
                    .Append(mlContext.Transforms.NormalizeMinMax("Humidity"))
                    .Append(mlContext.Transforms.NormalizeMinMax("Windspeed"))

                // Объединяем все признаки в один вектор
                    .Append(mlContext.Transforms.Concatenate("Features",
                        "Season",
                        "Month",
                        "Hour",
                        "Holiday",
                        "Weekday",
                        "WorkingDay",
                        "WeatherCondition",
                        "Temperature",
                        "Humidity",
                        "Windspeed"
                    ))

                // Обучение модели
                    .Append(mlContext.BinaryClassification.Trainers.FastTree(
                        labelColumnName: "RentalType",  // Используем boolean метку напрямую
                        featureColumnName: "Features"
                    ));

                // 5. Обучаем модель
                Console.WriteLine("Обучение модели...");
                var model = pipeline.Fit(splitData.TrainSet);

                // 6. Делаем предсказания на тестовой выборке
                Console.WriteLine("Выполняем оценку...");
                var predictions = model.Transform(splitData.TestSet);

                // 7. Оцениваем качество модели
                Console.WriteLine("Оценка качества модели...");
                var metrics = mlContext.BinaryClassification.Evaluate(predictions, labelColumnName: "RentalType");

                Console.WriteLine($"AUC: {metrics.AreaUnderRocCurve:F2}");
                Console.WriteLine($"F1 Score: {metrics.F1Score:F2}");

                // 8. Создаем движок предсказаний
                Console.WriteLine("Создаем движок предсказаний...");
                var predictionEngine = mlContext.Model.CreatePredictionEngine<BikeRentalData, RentalTypePrediction>(model);

                // Пример: зима, июнь, день, не выходной
                var example1 = new BikeRentalData()
                {
                    Season = 1,
                    Month = 6,
                    Hour = 12,
                    Holiday = 0,
                    Weekday = 3,
                    WorkingDay = 1,
                    WeatherCondition = 1,
                    Temperature = 18.0f,
                    Humidity = 75.0f,
                    Windspeed = 8.0f,
                    
                };

                var result1 = predictionEngine.Predict(example1);
                Console.WriteLine($"Пример: {example1.WeatherCondition} погода, темп {example1.Temperature}, " +
                                  $"предсказание: {result1.PredictedRentalType} (вероятность: {result1.Probability:F2})");

                // Пример 2: плохая погода
                var example2 = new BikeRentalData()
                {
                    Season = 4,
                    Month = 12,
                    Hour = 16,
                    Holiday = 0,
                    Weekday = 5,
                    WorkingDay = 1,
                    WeatherCondition = 4,
                    Temperature = -2.0f,
                    Humidity = 90.0f,
                    Windspeed = 20.0f,
                    
                };

                var result2 = predictionEngine.Predict(example2);
                Console.WriteLine($"Пример: {example2.WeatherCondition} погода, темп {example2.Temperature}, " +
                                  $"предсказание: {result2.PredictedRentalType} (вероятность: {result2.Probability:F2})");

                // 9. Завершение
                Console.WriteLine("\nНажмите любую клавишу для завершения...");
                Console.ReadKey();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Ошибка: {ex.Message}");
                Console.WriteLine($"Стек вызовов: {ex.StackTrace}");
                Console.WriteLine("Нажмите любую клавишу для завершения...");
                Console.ReadKey();
            }
        }
    }
}
