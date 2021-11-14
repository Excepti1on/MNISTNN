using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MNISTNN
{
    public static class ArrayExtensions
    {
        public static void ForEach<T>(this T[,] source, Action<int, int> action)
        {
            for (int i = 0; i < source.GetLength(0); i++)
            {
                for (int j = 0; j < source.GetLength(1); j++)
                {
                    action(i, j);
                }
            }
        }
    }
}
