using System;
using System.Collections.Generic;
using System.IO;

namespace MNISTNN
{
    public static class MNISTReader
    {
        private const string TrainImages = "F:\\repos\\MNISTNN\\MNISTNN\\mnist\\train-images.idx3-ubyte";
        private const string TrainLabels = "F:\\repos\\MNISTNN\\MNISTNN\\mnist\\train-labels.idx1-ubyte";
        private const string TestImages = "F:\\repos\\MNISTNN\\MNISTNN\\mnist\\t10k-images.idx3-ubyte";
        private const string TestLabels = "F:\\repos\\MNISTNN\\MNISTNN\\mnist\\t10k-labels.idx1-ubyte";

        public static IEnumerable<Image> ReadTrainData()
        {
            foreach (Image image in Read(TrainImages, TrainLabels))
            {
                yield return image;
            }
        }

        public static IEnumerable<Image> ReadTestData()
        {
            foreach (Image image in Read(TestImages, TestLabels))
            {
                yield return image;
            }

        }

        private static IEnumerable<Image> Read(string imagePath, string labelPath)
        {
            using BinaryReader labels = new BinaryReader(new FileStream(labelPath, FileMode.Open));
            using BinaryReader images = new BinaryReader(new FileStream(imagePath, FileMode.Open));

            int magicNumber = images.ReadBigInt32();
            int numImages = images.ReadBigInt32();
            int numRows = images.ReadBigInt32();
            int numCols = images.ReadBigInt32();

            int magicNumberLabel = labels.ReadBigInt32();
            int numLabels = labels.ReadBigInt32();

            for (int i = 0; i < numImages; i++)
            {
                byte[] bytes = images.ReadBytes(numRows * numCols);

                yield return new Image()
                {
                    Pixels = bytes,
                    Label = labels.ReadByte()
                };
            }
        }

        public static int ReadBigInt32(this BinaryReader br)
        {
            byte[] bytes = br.ReadBytes(sizeof(Int32));
            if (BitConverter.IsLittleEndian)
            {
                Array.Reverse(bytes);
            }
            return BitConverter.ToInt32(bytes, 0);
        }



    }
    public class Image
    {
        public byte[] Pixels { get; set; }
        public byte Label { get; set; }
    }
}