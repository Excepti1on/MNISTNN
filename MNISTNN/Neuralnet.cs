using System;
using System.Collections.Generic;
using System.Linq;

namespace MNISTNN
{
    class Neuralnet
    {
        private const int imageSize = 784;
        private const int layer1Size = 128;
        private const int layer2Size = 64;
        private const int labelSize = 10;

        private double[] input = new double[imageSize];

        private double[] layer1 = new double[layer1Size];
        private double[] layer1Sig = new double[layer1Size];
        private double[,] weightsl1 = new double[layer1Size, imageSize];
        private double[,] weightsGradientl1 = new double[layer1Size, imageSize];
        private double[] biasl1 = new double[layer1Size];
        private double[] biasGradientl1 = new double[layer1Size];

        private double[] layer2 = new double[layer2Size];
        private double[] layer2Sig = new double[layer2Size];
        private double[,] weightsl2 = new double[layer2Size, layer1Size];
        private double[,] weightsGradientl2 = new double[layer2Size, layer1Size];
        private double[] biasl2 = new double[layer2Size];
        private double[] biasGradientl2 = new double[layer2Size];

        private double[] layer3 = new double[labelSize];
        private double[,] weightsl3 = new double[labelSize, layer2Size];
        private double[,] weightsGradientl3 = new double[labelSize, layer2Size];
        private double[] biasl3 = new double[labelSize];
        private double[] biasGradientl3 = new double[labelSize];

        private double[] output = new double[labelSize];

        public Neuralnet()
        {
            RandomizeWeights(weightsl1);
            RandomizeWeights(weightsl2);
            RandomizeWeights(weightsl3);
            RandomizeBias(biasl1);
            RandomizeBias(biasl2);
            RandomizeBias(biasl3);
        }
        private double[] NormalizePixels(byte[] input)
        {
            double[] output = new double[input.Length];
            for (int i = 0; i < input.Length; i++)
            {
                output[i] = input[i] / 255d;
            }
            return output;
        }

        private void RandomizeWeights(double[,] weights)
        {
            Random _random = new();
            weights.ForEach((i, j) => weights[i, j] = _random.NextDouble());
        }
        private void RandomizeBias(double[] bias)
        {
            Random _random = new();
            for (int i = 0; i < bias.Length; i++)
            {
                bias[i] = _random.NextDouble();
            }
        }

        private void Forwards(Image image)
        {
            input = NormalizePixels(image.Pixels);

            layer1 = MathFunctions.MatrixDotVector(weightsl1, input);
            layer1 = layer1.VectorAdd(biasl1);
            layer1Sig = MathFunctions.SigmoidVec(layer1);

            layer2 = MathFunctions.MatrixDotVector(weightsl2, layer1Sig);
            layer2 = layer2.VectorAdd(biasl2);
            layer2Sig = MathFunctions.SigmoidVec(layer2);

            layer3 = MathFunctions.MatrixDotVector(weightsl3, layer2Sig);
            layer3 = layer3.VectorAdd(biasl3);
            output = MathFunctions.Softmax(layer3);
        }

        private double GradientDescent(Image image)
        {
            double loss;
            int i, j;
            double[] label = new double[labelSize];

            for (i = 0; i < labelSize; i++)
            {
                label[i] = (i == image.Label) ? 1 : 0;
            }
            loss = MathFunctions.CrossEntropy(output, label);

            //Layer3
            double[] error3 = MathFunctions.CrossEntropyDx(output, label);
            double[,] gradient3 = MathFunctions.VectorOuter(error3, layer2Sig);
            double bGradient3 = error3.Sum();
            for (i = 0; i < labelSize; i++)
            {
                for (j = 0; j < layer2Size; j++)
                {
                    weightsGradientl3[i, j] += gradient3[i, j];
                }
                biasGradientl3[i] += bGradient3;
            }

            //Layer2
            double[] error2 = MathFunctions.MatrixDotVector(MathFunctions.TransposeMatrix(weightsl3), error3);
            double[] sigmoid2dx = MathFunctions.SigmoidDxVec(layer2);
            double[] gradient2 = new double[layer2Size];
            for (i = 0; i < layer2Size; i++)
            {
                gradient2[i] = sigmoid2dx[i] * error2[i];
            }
            double[,] helper2 = MathFunctions.VectorOuter(gradient2, layer1Sig);
            double bGradient2 = error2.Sum();
            for (i = 0; i < layer2Size; i++)
            {
                for (j = 0; j < layer1Size; j++)
                {
                    weightsGradientl2[i, j] += helper2[i, j];
                }
                biasGradientl2[i] += bGradient2;
            }

            //Layer1
            double[] error1 = MathFunctions.MatrixDotVector(MathFunctions.TransposeMatrix(weightsl2), error2);
            double[] sigmoid1dx = MathFunctions.SigmoidDxVec(layer1);
            double[] gradient1 = new double[layer1Size];
            for (i = 0; i < layer1Size; i++)
            {
                gradient1[i] = sigmoid1dx[i] * error1[i];
            }
            double[,] helper1 = MathFunctions.VectorOuter(gradient1, input);
            double bGradient1 = error1.Sum();
            for (i = 0; i < layer1Size; i++)
            {
                for (j = 0; j < imageSize; j++)
                {
                    weightsGradientl1[i, j] += helper1[i, j];
                }
                biasGradientl1[i] += bGradient1;
            }
            return loss;
            
        }
        private void UpdateNetwork(double learningRate)
        {
            int i, j;
            for (i = 0; i < labelSize; i++)
            {
                for (j = 0; j < layer2Size; j++)
                {
                    weightsl3[i, j] -= weightsGradientl3[i, j] * learningRate;
                }
                biasl3[i] -= biasGradientl3[i] + learningRate;
            }
            for (i = 0; i < layer2Size; i++)
            {
                for (j = 0; j < layer1Size; j++)
                {
                    weightsl2[i, j] -= weightsGradientl2[i, j] * learningRate;
                }
                biasl2[i] -= biasGradientl2[i] * learningRate;
            }
            for (i = 0; i < layer1Size; i++)
            {
                for (j = 0; j < imageSize; j++)
                {
                    weightsl1[i, j] -= weightsGradientl1[i, j] * learningRate;
                }
                biasl1[i] -= biasGradientl1[i] * learningRate;
            }
        }
        public double CalculateAccuracy(IEnumerable<Image> images)
        {
            double predictions = 0;
            double prediction;
            foreach (Image _image in images)
            {
                Forwards(_image);
                prediction = Array.IndexOf(output, output.Max());
                predictions += (prediction == _image.Label) ? 1d : 0d;
            }
            return predictions / images.Count();
        }
        public double Train(IEnumerable<Image> images, double learningRate)
        {
            double loss = 0;
            foreach (Image image in images)
            {
                Forwards(image);
                //Console.WriteLine(Array.IndexOf(output, output.Max()) + "\t" + image.Label);
                loss += GradientDescent(image);
            }
            UpdateNetwork(learningRate);
            return loss;

        }

    }
}
