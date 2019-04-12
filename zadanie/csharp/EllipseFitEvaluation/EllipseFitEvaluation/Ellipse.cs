using System;
using System.IO;
using OpenCvSharp;

namespace EllipseFitEvaluation
{
    public class Ellipse
    {
        public Point2f Center;
        public Size2f Axes;
        public float Angle;

        public Ellipse(Point2f center, Size2f axes, float angle)
        {
            Center = center;
            Axes = axes;
            Angle = angle;
        }
    }

    public class GtEllipse : Ellipse
    {
        public Size ImageSize;

        public GtEllipse(Point2f center, Size2f axes, float angle, Size imageSize) : base(center, axes, angle)
        {
            ImageSize = imageSize;
        }
    }
}