using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace MNISTNN
{
    class Program
    {
        static double loss, accuracy;
        static void Main(string[] args)
        {
            var timer = new Stopwatch();
            timer.Start();
            int batchSize = 10;
            int epochs = 10000;
            int trainSize = 60000;
            int testSize = 15000;
            IEnumerable<Image> trainDataset = MNISTReader.ReadTrainData();
            IEnumerable<Image> testDataset = MNISTReader.ReadTestData();

            Neuralnet Network = new();

            for (int i = 0; i < epochs; i++)
            {
                IEnumerable<Image> batch = trainDataset.Skip(i * batchSize % (trainSize / batchSize)).Take(batchSize);
                loss = Network.Train(batch, 0.005);
                IEnumerable<Image> testBatch = testDataset.Skip(i * batchSize % (testSize / batchSize * 5)).Take(batchSize);
                accuracy = Network.CalculateAccuracy(testDataset);
                Console.WriteLine(loss + "\t" + accuracy);
            }
            timer.Stop();
            Console.WriteLine(timer.Elapsed);
            
            double[] vector = { 1, 2, 3 };
            double[,] matrix = { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } };
            foreach(var element in MathFunctions.MatrixDotVector(matrix, vector))
            {
                Console.WriteLine(element);
            }
        }
    }
}
