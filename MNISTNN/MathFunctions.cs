using System;
using System.Linq;


namespace MNISTNN
{
    public static class MathFunctions
    {
        public static double[] SigmoidVec(double[] input)
        {
            int l = input.Length;
            double[] output = new double[l];
            for (int i = 0; i < l; i++)
            {
                output[i] = Sigmoid(input[i]);
            }
            return output;
        }
        public static double[] SigmoidDxVec(double[] input)
        {
            int l = input.Length;
            double[] output = new double[l];
            for (int i = 0; i < l; i++)
            {
                output[i] = SigmoidDx(input[i]);
            }
            return output;
        }
        public static double Sigmoid(double input)
        {
            return 1 / (Math.Exp(-input) + 1d);
        }
        public static double SigmoidDx(double input)
        {
            return Sigmoid(input)*(1d-Sigmoid(input));
        }
        public static double[] Softmax(double[] input)
        {
            double sum;
            double max = input.Max();
            int l = input.Length;
            double[] output = new double[l];
            double[] helper = new double[l];
            for (int i = 0; i < l; i++)
            {
                helper[i] = Math.Exp(input[i] - max);
            }
            sum = helper.Sum();
            for (int i = 0; i < l; i++)
            {
                output[i] = helper[i] / sum;
            }
            return output;
        }
        public static double[,] SoftmaxDx(double[] input)
        {
            double[,] output = new double[input.Length, input.Length];
            double[] helper = Softmax(input);
            for (int i = 0; i < input.Length; i++)
            {
                for (int j = 0; j < input.Length; j++)
                {
                    if(i == j)
                    {
                        output[i, j] = helper[i] * (1 - helper[i]);
                    }
                    else
                    {
                        output[i, j] =-(helper[i] * helper[j]);
                    }
                }
            }
            return output;
        }
        public static double CrossEntropy(double[] prediction, double[] target)
        {
            double output = 0;
            for (int i = 0; i < prediction.Length; i++)
            {
                output += Math.Log(prediction[i]) * target[i];
                //Console.WriteLine(Math.Log(prediction[i]));
            }
            return output * -1d;
        }

        public static double[] CrossEntropyDx(double[] input, double[] label)
        {
            double[] output = new double[input.Length];
            for (int i = 0; i < input.Length; i++)
            {
                output[i] = input[i] - label[i];
            }
            return output;
            
        }
        public static double VectorDot(double[] vector1, double[] vector2)
        {
            int l;
            if (vector1.Length != vector2.Length)
            {
                throw new InvalidOperationException("Dimensions of the Vectors are not Equal");
                
            }
            l = vector1.Length;
            double output = 0;
            for (int i = 0; i < l; i++)
            {
                output += vector1[i] * vector2[i];
            }
            return output;
        }
        public static double[] VectorAdd(this double[] vector1, double[] vector2)
        {
            double[] output = new double[vector2.Length];
            for (int i = 0; i < vector1.Length; i++)
            {
                output[i] += vector2[i];
            }
            return output;
        }
        public static double[] MatrixDotVector(double[,] matrix, double[] vector)
        {

            if(vector.Length != matrix.GetLength(1))
            {
                throw new InvalidOperationException("Dimensions do not match");
            }
            int l1 = matrix.GetLength(0);
            int l2 = matrix.GetLength(1);
            double[] output = new double[l1];
            for (int i = 0; i < l1; i++)
            {
                for (int j = 0; j< l2; j++)
                {
                    output[i] += vector[j] * matrix[i, j];
                }
            }
            return output;
        }

        public static double[,] TransposeMatrix(double[,] matrix)
        {
            int l1 = matrix.GetLength(0);
            int l2 = matrix.GetLength(1);
            double[,] output = new double[l2, l1];
            for (int i = 0; i < l1; i++)
            {
                for (int j = 0; j < l2; j++)
                {
                    output[j, i] = matrix[i, j];
                }
            }
            return output;
        }
        public static double[,] VectorOuter(double[] vector1, double[] vector2)
        {
            int l1 = vector1.Length;
            int l2 = vector2.Length;
            double[,] output = new double[l1, l2];
            for (int i = 0; i < l1; i++)
            {
                for (int j = 0; j < l2; j++)
                {
                    output[i, j] = vector1[i] * vector2[j];
                }
            }
            return output;
        }
    }

}
