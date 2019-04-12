using System;
using System.IO;
using OpenCvSharp;

namespace EllipseFitEvaluation
{
    public class EllipseFitEvaluation
    {
        /// <summary>
        /// Evaluate ellipse fit by comparing its parameters to the ground truth ellipse's parameters from the CSV file.
        /// </summary>
        /// <param name="imageFilename">Filename of the image</param>
        /// <param name="fitEllipse">Fitted ellipse's parameters</param>
        /// <param name="csvFilepath">Filepath to the CSV file with ground truth ellipses' parameters</param>
        /// <returns>Fitted ellipse's fit score</returns>
        public static float EvaluateEllipseFit(string imageFilename, Ellipse fitEllipse, string csvFilepath)
        {
            GtEllipse gtEllipse = GetGtEllipseFromCsv(imageFilename, csvFilepath);

            if (gtEllipse != null)
            {
                if (fitEllipse != null)
                {
                    return GetEllipseFitScore(fitEllipse, gtEllipse);
                }
                else
                {
                    return 0.0f;
                }
            }
            else
            {
                if (fitEllipse != null)
                {
                    return 0.0f;
                }
                else
                {
                    return 1.0f;
                }
            }
        }

        /// <summary>
        /// Return the ground truth ellipse's parameters from the CSV file.
        /// </summary>
        /// <param name="imageFilename">Filename of the image</param>
        /// <param name="csvFilepath">Filepath to the CSV file with ground truth ellipses' parameters</param>
        /// <returns>Ground truth ellipse's parameters</returns>
        private static GtEllipse GetGtEllipseFromCsv(string imageFilename, string csvFilepath)
        {
            using (StreamReader sr = new StreamReader(csvFilepath))
            {
                GtEllipse gtEllipse;
                string line;
                sr.ReadLine();  // skip the header
                while ((line = sr.ReadLine()) != null)
                {
                    string[] columns = line.Split(',');
                    if (string.Compare(columns[0], imageFilename) == 0)
                    {
                        if (string.IsNullOrEmpty(columns[1]))
                        {
                            gtEllipse = null;
                        }
                        else
                        {
                            Point2f center = new Point2f(Convert.ToSingle(columns[1]), Convert.ToSingle(columns[2]));
                            Size2f axes = new Size2f(Convert.ToSingle(columns[3]), Convert.ToSingle(columns[4]));
                            float angle = Convert.ToSingle(columns[5]);
                            Size imageSize = new Size(Convert.ToInt32(columns[6]), Convert.ToInt32(columns[7]));

                            gtEllipse = new GtEllipse(center, axes, angle, imageSize);
                        }
                        return gtEllipse;
                    };
                }
                throw new Exception("Filename not found in the CSV file.");
            }
        }

        /// <summary>
        /// Calculate ellipse's fit score based on fitted and ground truth ellipses' parameters.
        /// </summary>
        /// <param name="fitEllipse">Fitted ellipse's parameters</param>
        /// <param name="gtEllipse">Ground truth ellipse's parameters</param>
        /// <returns>Fitted ellipse's fit score</returns>
        private static float GetEllipseFitScore(Ellipse fitEllipse, GtEllipse gtEllipse)
        {
            Mat fitEllipseImage = DrawEllipse(fitEllipse, gtEllipse.ImageSize);
            Mat gtEllipseImage = DrawEllipse(gtEllipse, gtEllipse.ImageSize);
            Cv2.WaitKey();

            return EvaluateOverlap(gtEllipseImage, fitEllipseImage);
        }

        /// <summary>
        /// Creates a binary image containing an ellipse.
        /// </summary>
        /// <param name="ellipse">Ellipse's parameters</param>
        /// <param name="size">Image's size</param>
        /// <returns>Binary image containing an ellipse</returns>
        private static Mat DrawEllipse(Ellipse ellipse, Size size)
        {
            Mat ellipseImage = Mat.Zeros(size, MatType.CV_8U);
            Point center = new Point((int)Math.Round(ellipse.Center.X), (int)Math.Round(ellipse.Center.Y));
            Size axes = new Size((int)Math.Round(ellipse.Axes.Width), (int)Math.Round(ellipse.Axes.Height));
            double angle = ellipse.Angle;

            Cv2.Ellipse(ellipseImage, center, axes, angle, 0, 360, 1, 1);
            Cv2.Ellipse(ellipseImage, center, axes, angle, 0, 360, 1, Cv2.FILLED);

            return ellipseImage;
        }

        /// <summary>
        /// Calculate overlap of two binary images of ellipses.
        /// </summary>
        /// <param name="gtEllipseImage">Binary image with the ground truth ellipse</param>
        /// <param name="fitEllipseImage">Binary image with the fitted ellipse</param>
        /// <returns>Relative overlap of two binary images of ellipses</returns>
        private static float EvaluateOverlap(Mat gtEllipseImage, Mat fitEllipseImage)
        {
            int fitSum = (int)Cv2.Sum(fitEllipseImage).Val0;
            int gtSum = (int)Cv2.Sum(gtEllipseImage).Val0;

            Mat overlapImage = new Mat();
            Cv2.BitwiseAnd(fitEllipseImage, gtEllipseImage, overlapImage);
            int overlapSum = (int)Cv2.Sum(overlapImage).Val0;

            return overlapSum / (float)Math.Max(fitSum, gtSum);
        }
    }
}
